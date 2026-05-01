import torch
import random
import inspect

from tqdm import tqdm
from torch import Tensor
from typing import Optional
from torchmetrics import MeanMetric

from boltz.data import const
from boltz.model.model import Boltz1
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.confidence import confidence_loss
from boltz.model.optim.scheduler import AlphaFoldLRScheduler
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
)

from afgen.writer import Writer
from afgen.sequence import SequenceModule
from afgen.diffusion import AtomDiffusion
from afgen.inference import BatchUpdater


class AFGenModel(Boltz1):
    def __init__(
        self,
        sequence_prediction=False,
        sequence_model_args={},
        sequence_imitate_trunk=False,
        compile_sequence=False,
        no_atom_encoder=False,
        batch_updater: BatchUpdater=None,
        writer=None,
        **kwargs,
    ):
        cls_init_args_name = inspect.signature(super().__init__).parameters.keys()
        kwargs = {k: v for k, v in kwargs.items() if k in cls_init_args_name}
        kwargs['steering_args'] = {}
        super().__init__(**kwargs)

        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_z": kwargs['token_z'],
                "token_s": kwargs['token_s'],
                "atom_z": kwargs['atom_z'],
                "atom_s": kwargs['atom_s'],
                "atoms_per_window_queries": kwargs['atoms_per_window_queries'],
                "atoms_per_window_keys": kwargs['atoms_per_window_keys'],
                "atom_feature_dim": kwargs['atom_feature_dim'],
                **kwargs['score_model_args'],
            },
            compile_score=kwargs['compile_structure'],
            accumulate_token_repr=kwargs['confidence_model_args'].get("use_s_diffusion", False),
            accumulate_token_repr_seq=sequence_model_args.get("use_s_diffusion", False),
            **kwargs['diffusion_process_args'],
        )

        self.sequence_prediction = sequence_prediction
        if self.sequence_prediction:
            full_embedder_args = {
                "atom_s": kwargs['atom_s'],
                "atom_z": kwargs['atom_z'],
                "token_s": kwargs['token_s'],
                "token_z": kwargs['token_z'],
                "atoms_per_window_queries": kwargs['atoms_per_window_queries'],
                "atoms_per_window_keys": kwargs['atoms_per_window_keys'],
                "atom_feature_dim": kwargs['atom_feature_dim'],
                "no_atom_encoder": no_atom_encoder,
                **kwargs['embedder_args'],
            }
    
            self.sequence_module = SequenceModule(
                kwargs['token_s'],
                kwargs['token_z'],
                imitate_trunk=sequence_imitate_trunk,
                pairformer_args=kwargs['pairformer_args'],
                full_embedder_args=full_embedder_args,
                msa_args=kwargs['msa_args'],
                **sequence_model_args,
            )
            if compile_sequence:
                self.sequence_module = torch.compile(self.sequence_module)

            self.train_acc_logger = MeanMetric()
            self.train_mask_seq_rate_logger = MeanMetric()
            self.train_mask_msa_rate_logger = MeanMetric()
            self.val_seq_loss_logger = MeanMetric()
            self.val_mask_seq_rate_logger = MeanMetric()
            self.val_mask_msa_rate_logger = MeanMetric()
            self.val_seq_acc_logger = MeanMetric()
        
        # if self.training:
        #     assert self.structure_prediction_training + self.confidence_prediction \
        #         + self.sequence_prediction == 1, "Only one of the tasks should be trained"
        # Remove grad from weights they are not trained for ddp
        for name, param in self.named_parameters():
            if self.structure_prediction_training:
                if name.startswith('confidence_module'):
                    param.requires_grad = False
                elif name.startswith('sequence_module'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif self.confidence_prediction:
                if name.startswith("confidence_module"):
                    param.requires_grad = True
                elif name.startswith("structure_module.out_token_feat_update"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif self.sequence_prediction:
                if name.startswith("sequence_module"):
                    param.requires_grad = True
                elif name.startswith("structure_module.out_token_feat_update_seq"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.batch_updater = batch_updater
        self.writer: Writer = writer
    
    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        dict_out = {}

        # Compute input embeddings
        with torch.set_grad_enabled(self.training and self.structure_prediction_training):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Fixes an issue with unused parameters in autocast
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()

                    # Apply recycling
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # Compute pairwise stack
                    if not self.no_msa:
                        z = z + self.msa_module(z, s_inputs, feats)

                    # Revert to uncompiled version for validation
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}

        # Compute structure module
        if self.training and self.structure_prediction_training:
            dict_out.update(
                self.structure_module(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    multiplicity=multiplicity_diffusion_train,
                )
            )

        if (not self.training) or self.confidence_prediction or self.sequence_prediction:
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"],
                    multiplicity=diffusion_samples,
                    train_accumulate_token_repr=self.training,
                    train_accumulate_token_repr_seq=self.training,
                )
            )

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    s_diffusion=(
                        dict_out["diff_token_repr"]
                        if self.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                )
            )
            dict_out.pop("diff_token_repr", None)

        if self.sequence_prediction:
            dict_out.update(
                self.sequence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    s_diffusion=(
                        dict_out["diff_token_repr_sequence"]
                        if self.sequence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    multiplicity=diffusion_samples,
                )
            )
            dict_out.pop("diff_token_repr_sequence", None)

        return dict_out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # Sample recycling steps
        recycling_steps = random.randint(0, self.training_args.recycling_steps)

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=self.training_args.sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
            diffusion_samples=self.training_args.diffusion_samples,
        )

        # Compute losses
        if self.structure_prediction_training:
            disto_loss, _ = distogram_loss(
                out,
                batch,
            )
            try:
                diffusion_loss_dict = self.structure_module.compute_loss(
                    batch,
                    out,
                    multiplicity=self.training_args.diffusion_multiplicity,
                    **self.diffusion_loss_args,
                )
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                return None
        else:
            disto_loss = 0.0
            diffusion_loss_dict = {"loss": 0.0, "loss_breakdown": {}}

        if self.confidence_prediction:
            # confidence model symmetry correction
            true_coords, _, _, true_coords_resolved_mask = self.get_true_coordinates(
                batch,
                out,
                diffusion_samples=self.training_args.diffusion_samples,
                symmetry_correction=self.training_args.symmetry_correction,
            )
            confidence_loss_dict = confidence_loss(
                out,
                batch,
                true_coords,
                true_coords_resolved_mask,
                alpha_pae=self.alpha_pae,
                multiplicity=self.training_args.diffusion_samples,
            )
        else:
            confidence_loss_dict = {
                "loss": torch.tensor(0.0).to(batch["token_index"].device),
                "loss_breakdown": {},
            }

        if self.sequence_prediction:
            seq_loss, seq_acc, avg_seq_mask, avg_msa_mask = self.sequence_module.sequence_loss(
                out=out, batch=batch, multiplicity=self.training_args.diffusion_samples)
        else:
            seq_loss = torch.tensor(0.).to(batch["token_index"].device)
            seq_acc = 0.
            avg_seq_mask = 0.
            avg_msa_mask = 0.

        # Aggregate losses
        loss = (
            self.training_args.confidence_loss_weight * confidence_loss_dict["loss"]
            + self.training_args.diffusion_loss_weight * diffusion_loss_dict["loss"]
            + self.training_args.distogram_loss_weight * disto_loss
            + self.training_args.sequence_loss_weight * seq_loss
        )
        # Log losses
        if self.structure_prediction_training:
            self.log("train/distogram_loss", disto_loss)
            self.log("train/diffusion_loss", diffusion_loss_dict["loss"])
            for k, v in diffusion_loss_dict["loss_breakdown"].items():
                self.log(f"train/{k}", v)

        if self.confidence_prediction:
            self.train_confidence_loss_logger.update(
                confidence_loss_dict["loss"].detach()
            )

            for k in self.train_confidence_loss_dict_logger.keys():
                self.train_confidence_loss_dict_logger[k].update(
                    confidence_loss_dict["loss_breakdown"][k].detach()
                    if torch.is_tensor(confidence_loss_dict["loss_breakdown"][k])
                    else confidence_loss_dict["loss_breakdown"][k]
                )
        
        if self.sequence_prediction:
            self.train_acc_logger.update(seq_acc, weight=torch.sum(batch['mask_target']!=0))
            self.train_mask_seq_rate_logger.update(avg_seq_mask)
            self.train_mask_msa_rate_logger.update(avg_msa_mask)
            self.log(f"train/sequence_acc", self.train_acc_logger.compute())
            self.log(f"train/mask_seq_rate", self.train_mask_seq_rate_logger.compute())
            self.log(f"train/mask_msa_rate", self.train_mask_msa_rate_logger.compute())

        self.log("train/loss", loss)
        self.training_log()
        return loss

    def training_log(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False)

        self.log(
            "train/grad_norm_structure_module",
            self.gradient_norm(self.structure_module),
            prog_bar=False,
        )
        self.log(
            "train/param_norm_structure_module",
            self.parameter_norm(self.structure_module),
            prog_bar=False,
        )
        
        if self.structure_prediction_training:
            self.log(
            "train/grad_norm_msa_module",
            self.gradient_norm(self.msa_module),
            prog_bar=False,
            )
            self.log(
                "train/param_norm_msa_module",
                self.parameter_norm(self.msa_module),
                prog_bar=False,
            )
            self.log(
                "train/grad_norm_pairformer_module",
                self.gradient_norm(self.pairformer_module),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_pairformer_module",
                self.parameter_norm(self.pairformer_module),
                prog_bar=False,
            )
        
        if self.confidence_prediction:
            self.log(
                "train/grad_norm_confidence_module",
                self.gradient_norm(self.confidence_module),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_confidence_module",
                self.parameter_norm(self.confidence_module),
                prog_bar=False,
            )

        if self.sequence_prediction:
            self.log(
                "train/grad_norm_sequence_module",
                self.gradient_norm(self.sequence_module),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_sequence_module",
                self.parameter_norm(self.sequence_module),
                prog_bar=False,
            )

    def validation_step(self, batch, batch_idx):
        n_samples = self.validation_args.diffusion_samples
        out = self(
            batch,
            recycling_steps=self.validation_args.recycling_steps,
            num_sampling_steps=self.validation_args.sampling_steps,
            diffusion_samples=n_samples,
            run_confidence_sequentially=self.validation_args.run_confidence_sequentially,
        )

        # Compute distogram LDDT
        boundaries = torch.linspace(2, 22.0, 63)
        lower = torch.tensor([1.0])
        upper = torch.tensor([22.0 + 5.0])
        exp_boundaries = torch.cat((lower, boundaries, upper))
        mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
            out["pdistogram"]
        )

        # Compute predicted dists
        preds = out["pdistogram"]
        pred_softmax = torch.softmax(preds, dim=-1)
        pred_softmax = pred_softmax.argmax(dim=-1)
        pred_softmax = torch.nn.functional.one_hot(
            pred_softmax, num_classes=preds.shape[-1]
        )
        pred_dist = (pred_softmax * mid_points).sum(dim=-1)
        true_center = batch["disto_center"]
        true_dists = torch.cdist(true_center, true_center)

        # Compute lddt's
        batch["token_disto_mask"] = batch["token_disto_mask"]
        disto_lddt_dict, disto_total_dict = factored_token_lddt_dist_loss(
            feats=batch,
            true_d=true_dists,
            pred_d=pred_dist,
        )

        true_coords, rmsds, best_rmsds, true_coords_resolved_mask = (
            self.get_true_coordinates(
                batch=batch,
                out=out,
                diffusion_samples=n_samples,
                symmetry_correction=self.validation_args.symmetry_correction,
            )
        )

        all_lddt_dict, all_total_dict = factored_lddt_loss(
            feats=batch,
            atom_mask=true_coords_resolved_mask,
            true_atom_coords=true_coords,
            pred_atom_coords=out["sample_atom_coords"],
            multiplicity=n_samples,
        )

        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        B = true_coords.shape[0] // n_samples
        if n_samples > 1:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict.keys():
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7
            best_complex_idx = complex_lddt.reshape(-1, n_samples).argmax(dim=1)
            for key in all_lddt_dict:
                best_idx = all_lddt_dict[key].reshape(-1, n_samples).argmax(dim=1)
                best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                best_total_dict[key] = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_complex_idx
                ]
                best_complex_total_dict[key] = all_total_dict[key].reshape(
                    -1, n_samples
                )[torch.arange(B), best_complex_idx]
        else:
            best_lddt_dict = all_lddt_dict
            best_total_dict = all_total_dict
            best_complex_lddt_dict = all_lddt_dict
            best_complex_total_dict = all_total_dict

        # Filtering based on confidence
        if self.confidence_prediction and n_samples > 1:
            # note: for now we don't have pae predictions so have to use pLDDT instead of pTM
            # also, while AF3 differentiates the best prediction per confidence type we are currently not doing it
            # consider this in the future as well as weighing the different pLLDT types before aggregation
            mae_plddt_dict, total_mae_plddt_dict = compute_plddt_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_lddt=out["plddt"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pde_dict, total_mae_pde_dict = compute_pde_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pde=out["pde"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pae_dict, total_mae_pae_dict = compute_pae_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pae=out["pae"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )

            plddt = out["complex_plddt"].reshape(-1, n_samples)
            top1_idx = plddt.argmax(dim=1)
            iplddt = out["complex_iplddt"].reshape(-1, n_samples)
            iplddt_top1_idx = iplddt.argmax(dim=1)
            pde = out["complex_pde"].reshape(-1, n_samples)
            pde_top1_idx = pde.argmin(dim=1)
            ipde = out["complex_ipde"].reshape(-1, n_samples)
            ipde_top1_idx = ipde.argmin(dim=1)
            ptm = out["ptm"].reshape(-1, n_samples)
            ptm_top1_idx = ptm.argmax(dim=1)
            iptm = out["iptm"].reshape(-1, n_samples)
            iptm_top1_idx = iptm.argmax(dim=1)
            ligand_iptm = out["ligand_iptm"].reshape(-1, n_samples)
            ligand_iptm_top1_idx = ligand_iptm.argmax(dim=1)
            protein_iptm = out["protein_iptm"].reshape(-1, n_samples)
            protein_iptm_top1_idx = protein_iptm.argmax(dim=1)

            for key in all_lddt_dict:
                top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                iplddt_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                iplddt_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                pde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                pde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                ipde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ipde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                ptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                ligand_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                ligand_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                protein_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]
                protein_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]

                self.top1_lddt[key].update(top1_lddt, top1_total)
                self.iplddt_top1_lddt[key].update(iplddt_top1_lddt, iplddt_top1_total)
                self.pde_top1_lddt[key].update(pde_top1_lddt, pde_top1_total)
                self.ipde_top1_lddt[key].update(ipde_top1_lddt, ipde_top1_total)
                self.ptm_top1_lddt[key].update(ptm_top1_lddt, ptm_top1_total)
                self.iptm_top1_lddt[key].update(iptm_top1_lddt, iptm_top1_total)
                self.ligand_iptm_top1_lddt[key].update(
                    ligand_iptm_top1_lddt, ligand_iptm_top1_total
                )
                self.protein_iptm_top1_lddt[key].update(
                    protein_iptm_top1_lddt, protein_iptm_top1_total
                )

                self.avg_lddt[key].update(all_lddt_dict[key], all_total_dict[key])
                self.pde_mae[key].update(mae_pde_dict[key], total_mae_pde_dict[key])
                self.pae_mae[key].update(mae_pae_dict[key], total_mae_pae_dict[key])

            for key in mae_plddt_dict:
                self.plddt_mae[key].update(
                    mae_plddt_dict[key], total_mae_plddt_dict[key]
                )

        for m in const.out_types:
            if m == "ligand_protein":
                if torch.any(
                    batch["pocket_feature"][
                        :, :, const.pocket_contact_info["POCKET"]
                    ].bool()
                ):
                    self.lddt["pocket_ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["pocket_ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["pocket_ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
                else:
                    self.lddt["ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
            else:
                self.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
                self.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
                self.complex_lddt[m].update(
                    best_complex_lddt_dict[m], best_complex_total_dict[m]
                )
        self.rmsd.update(rmsds)
        self.best_rmsd.update(best_rmsds)

        # Compute sequence metrics
        if self.sequence_prediction:
            seq_loss, seq_acc, avg_seq_mask, avg_msa_mask = self.sequence_module.sequence_loss(
                out=out, batch=batch, multiplicity=n_samples
            )
            self.val_seq_loss_logger.update(seq_loss.detach())
            self.val_seq_acc_logger.update(seq_acc.detach())
            self.val_mask_seq_rate_logger.update(avg_seq_mask.detach())
            self.val_mask_msa_rate_logger.update(avg_msa_mask.detach())

    def on_validation_epoch_end(self):
        self.log("val/sequence_loss", self.val_seq_loss_logger, prog_bar=False)
        self.log("val/sequence_acc", self.val_seq_acc_logger, prog_bar=False)
        self.log("val/mask_seq_rate", self.val_mask_seq_rate_logger, prog_bar=False)
        self.log("val/mask_msa_rate", self.val_mask_msa_rate_logger, prog_bar=False)
        super().on_validation_epoch_end()

    def configure_optimizers(self):
        """Configure the optimizer."""
        parameters = [
            p for p in self.parameters() if p.requires_grad
        ]

        optimizer = torch.optim.Adam(
            parameters,
            betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
            eps=self.training_args.adam_eps,
            lr=self.training_args.base_lr,
        )
        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        if not self.predict_args['predict_structure']:
            return self.predict_sequence(batch)
        else:
            return self.predict_structure(batch)

    def predict_sequence(self, batch):
        assert batch['res_type'].size(0) == 1, "only support batch size 1 now"

        temperature = self.predict_args["temperature"]
        sym_id = batch['sym_id']
        entity_id = batch['entity_id']
        res_idx = batch['residue_index']
        res_type = torch.argmax(batch['res_type'], dim=-1)
        B, N = res_type.size()

        # only unmask chains with sym_id being 0
        seq_mask = (res_type == const.unk_token_ids['PROTEIN']) & \
            (batch['mol_type'] == const.chain_type_ids['PROTEIN']) & \
                (batch['sym_id'] == 0)      # B, N_token

        tot_unmask_num = torch.sum(seq_mask, dim=-1)      # B
        unmask_per_step = calculate_unmask_per_step(tot_unmask_num)
        assert (torch.sum(unmask_per_step, dim=-1) == tot_unmask_num).all()
        
        sequence_step = unmask_per_step.size(1)
        # iteratively update sequence
        for idx in tqdm(range(sequence_step+1)):            
            if idx < sequence_step:
                diffusion_samples = 1
            else:
                diffusion_samples=self.predict_args["diffusion_samples"]

            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=True,
            )
            
            if idx < sequence_step:
                # sample sequence
                logits = out['s_pred'] / temperature     # B, N_token, 33
                probs = torch.softmax(logits, dim=-1)

                # random unmask strategy
                # unmask_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
                # for i in range(B):
                #     true_indices = torch.nonzero(seq_mask[i], as_tuple=False).squeeze(-1)
                    
                #     if len(true_indices) <= unmask_per_step[i, idx]:
                #         selected_indices = true_indices
                #     else:
                #         selected_indices = true_indices[torch.randperm(len(true_indices))[:unmask_per_step[i, idx]]]
                #     unmask_mask[i, selected_indices] = True

                # structure aware unmask strategy
                unmask_mask = structure_aware_unmask(
                    seq_mask,
                    num=unmask_per_step[:, idx],
                    x_pred=out['sample_atom_coords'],
                    token_to_rep_atom=batch['token_to_rep_atom'],
                )
                assert (res_type[unmask_mask] == const.unk_token_ids['PROTEIN']).all()

                sampled_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B, N)
                assert (sampled_token[unmask_mask] != const.unk_token_ids['PROTEIN']).all()
                
                res_type = torch.where(unmask_mask, sampled_token, res_type)

                # unmask homo chains
                unmask_indices = unmask_mask.nonzero(as_tuple=False)  # (K, 2)
                for batch_idx, seq_idx in unmask_indices:
                    current_entity_id = entity_id[batch_idx, seq_idx]
                    current_res_idx = res_idx[batch_idx, seq_idx]
                    current_res_type = int(res_type[batch_idx, seq_idx])

                    homo_mask = (entity_id[batch_idx] == current_entity_id) & (res_idx[batch_idx] == current_res_idx)
                    res_type[batch_idx, homo_mask] = current_res_type                
                
                seq_mask = (res_type == const.unk_token_ids['PROTEIN']) & \
                    (batch['mol_type'] == const.chain_type_ids['PROTEIN']) & \
                        (batch['sym_id'] == 0)      # B, N_token

                # update batch sequence
                batch = self.batch_updater.update_batch(batch, res_type=res_type, target="residue")
                
            else:
                prot_mask = batch['mol_type'] == const.chain_type_ids['PROTEIN']
                assert (res_type[prot_mask] != const.unk_token_ids['PROTEIN']).all()

            pred_dict = self.make_out(batch, out)
            self.writer.write(pred_dict, batch=batch, sub_dir=f"{idx}")
            
            if idx < sequence_step:
                # update strucure atom and atom features
                batch = self.batch_updater.update_batch(batch, target="atom", res_type=res_type, unmask_mask=unmask_mask)
            
            if (idx == sequence_step-1) and self.predict_args['update_msa']:
                batch = self.batch_updater.update_batch(batch, target="msa", msa_dir=self.writer.out_dir/f"{idx+1}"/"msas")

        return
    
    def predict_structure(self, batch):
        assert not self.sequence_prediction, "should not turn on sequence module during strucutre preidiction"
        out = self(
            batch,
            recycling_steps=self.predict_args["recycling_steps"],
            num_sampling_steps=self.predict_args["sampling_steps"],
            diffusion_samples=self.predict_args["diffusion_samples"],
            run_confidence_sequentially=True,
        )
        pred_dict = self.make_out(batch, out)
        self.writer.write(pred_dict, batch=batch)

        return

    def make_out(self, batch, out):
        pred_dict = {}
        pred_dict["masks"] = batch["atom_pad_mask"]
        pred_dict["coords"] = out["sample_atom_coords"]
        if self.predict_args.get("write_confidence_summary", True):
            pred_dict["confidence_score"] = (
                4 * out["complex_plddt"] +
                (out["iptm"] if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"])) else out["ptm"])
            ) / 5
            for key in [
                "ptm",
                "iptm",
                "ligand_iptm",
                "protein_iptm",
                "pair_chains_iptm",
                "complex_plddt",
                "complex_iplddt",
                "complex_pde",
                "complex_ipde",
                "plddt",
            ]:
                pred_dict[key] = out[key]
        if self.predict_args.get("write_full_pae", True):
            pred_dict["pae"] = out["pae"]
        if self.predict_args.get("write_full_pde", False):
            pred_dict["pde"] = out["pde"]
        return pred_dict


# def calculate_unmask_per_step(tot_unmask_num):
#     B = tot_unmask_num.size(0)
#     device = tot_unmask_num.device

#     unmask_pattern = [4, 4, 8, 8, 16, 20]
#     # unmask_pattern = [50, 50, 100, 100]
#     unmask_per_step = []

#     step = 0
#     while True:
#         cur_num = unmask_pattern[-1 if step >= len(unmask_pattern) else step]
#         _unmask_step = torch.LongTensor([cur_num] * B).to(device)
#         _unmask_step = torch.where(
#             tot_unmask_num > _unmask_step, _unmask_step, tot_unmask_num 
#         )
#         unmask_per_step.append(_unmask_step)
#         tot_unmask_num = tot_unmask_num - _unmask_step
#         if torch.sum(tot_unmask_num) == 0:
#             break
#         step += 1

#     unmask_per_step = torch.stack(unmask_per_step, dim=1)
#     return unmask_per_step


import numpy as np
def calculate_unmask_per_step(tot_unmask_num):
    B = tot_unmask_num.size(0)
    device = tot_unmask_num.device
    assert B == 1, "only support batch size equal to 1"

    # if tot_unmask_num[0] > 50:
    unmask_pattern = np.array([2, 5, 10, 15, 15, 15, 15, 15, 8])
    # else:
    #     unmask_pattern = np.array([1, 5, 10, 10, 10, 10, 4])
    assert np.sum(unmask_pattern) == 100
    unmask_pattern = unmask_pattern / 100.
    
    unmask_per_step = []
    cur_unmask_num = 0
    for step in range(len(unmask_pattern)):
        if step == len(unmask_pattern) - 1:
            _unmask_step = tot_unmask_num - cur_unmask_num
        else:
            ratio = float(unmask_pattern[step])
            _unmask_step = torch.round(ratio * tot_unmask_num).long()
        cur_unmask_num += _unmask_step
        unmask_per_step.append(_unmask_step)
        assert (_unmask_step != 0).all()

    unmask_per_step = torch.stack(unmask_per_step, dim=1)
    return unmask_per_step


def structure_aware_unmask(seq_mask, num, x_pred, token_to_rep_atom, min_dist=5.0):
    """To determine the unmask order

    Args:
        seq_mask (torch.BoolTensor): (B, N), indicates mask position
        num (torch.LongTensor): (B, ), num to unmask
        x_pred (torch.FloatTensor): _description_
        token_to_rep_atom (torch.LongTensor): _description_
        min_dist (float, optional): _description_. Defaults to 5.0.
    """
    x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
    cmap = torch.cdist(x_pred_repr, x_pred_repr) <= min_dist

    B, N, _ = cmap.shape
    degree = cmap.sum(dim=-1)

    masked_degree = degree.masked_fill(~seq_mask, -1)
    topk_vals, topk_idx = torch.topk(masked_degree, k=seq_mask.size(1), dim=-1)
    unmask_mask = torch.zeros_like(seq_mask, dtype=torch.bool)

    for b in range(B):
        k = num[b].item()
        selected_idx = topk_idx[b, :k]
        unmask_mask[b, selected_idx] = True

    return unmask_mask