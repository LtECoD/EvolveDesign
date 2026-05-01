import torch
from torch import nn
import torch.nn.functional as F

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import LinearNoBias


class SequenceModule(nn.Module):
    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        use_s_diffusion=False,
        add_z_input_to_z=False,
        imitate_trunk=False,
        full_embedder_args: dict = None,
        msa_args: dict = None,
        compile_pairformer=False,
    ):
        super().__init__()
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )

        self.use_s_diffusion = use_s_diffusion
        if use_s_diffusion:
            self.s_diffusion_norm = nn.LayerNorm(2 * token_s)
            self.s_diffusion_to_s = LinearNoBias(2 * token_s, token_s)
            init.gating_init_(self.s_diffusion_to_s.weight)

        self.s_to_z = LinearNoBias(s_input_dim, token_z)
        self.s_to_z_transpose = LinearNoBias(s_input_dim, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        self.imitate_trunk = imitate_trunk
        if self.imitate_trunk:
            s_input_dim = (
                token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
            )
            self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
            self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
            self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

            # Input embeddings
            self.input_embedder = InputEmbedder(**full_embedder_args)
            self.rel_pos = RelativePositionEncoder(token_z)
            self.token_bonds = nn.Linear(1, token_z, bias=False)

            # Normalization layers
            self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            # Recycling projections
            self.s_recycle = nn.Linear(token_s, token_s, bias=False)
            self.z_recycle = nn.Linear(token_z, token_z, bias=False)
            init.gating_init_(self.s_recycle.weight)
            init.gating_init_(self.z_recycle.weight)

            # Pairwise stack
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
            self.pairformer_module = PairformerModule(
                token_s,
                token_z,
                **pairformer_args,
            )
            if compile_pairformer:
                # Big models hit the default cache limit (8)
                self.is_pairformer_compiled = True
                torch._dynamo.config.cache_size_limit = 512
                torch._dynamo.config.accumulated_cache_size_limit = 512
                self.pairformer_module = torch.compile(
                    self.pairformer_module,
                    dynamic=False,
                    fullgraph=False,
                )

            self.final_s_norm = nn.LayerNorm(token_s)
        else:
            self.s_inputs_norm = nn.LayerNorm(s_input_dim)
            if not self.no_update_s:
                self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            self.add_s_input_to_s = add_s_input_to_s
            if add_s_input_to_s:
                self.s_input_to_s = LinearNoBias(s_input_dim, token_s)
                init.gating_init_(self.s_input_to_s.weight)

            self.add_z_input_to_z = add_z_input_to_z
            if add_z_input_to_z:
                self.rel_pos = RelativePositionEncoder(token_z)
                self.token_bonds = nn.Linear(1, token_z, bias=False)

            self.pairformer_stack = PairformerModule(
                token_s,
                token_z,
                **pairformer_args,
            )

        self.sequence_head = LinearNoBias(token_s, const.num_tokens)
    
    def forward(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        feats,
        multiplicity=1,
        s_diffusion=None,
    ):
        if self.imitate_trunk:
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

            # Apply recycling
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))

        else:
            s_inputs = self.s_inputs_norm(s_inputs)
            if not self.no_update_s:
                s = self.s_norm(s)

            if self.add_s_input_to_s:
                s = s + self.s_input_to_s(s_inputs)

            z = self.z_norm(z)

            if self.add_z_input_to_z:
                relative_position_encoding = self.rel_pos(feats)
                z = z + relative_position_encoding
                z = z + self.token_bonds(feats["token_bonds"].float())

        s = s.repeat_interleave(multiplicity, 0)

        if self.use_s_diffusion:
            assert s_diffusion is not None
            s_diffusion = self.s_diffusion_norm(s_diffusion)
            s = s + self.s_diffusion_to_s(s_diffusion)

        z = z.repeat_interleave(multiplicity, 0)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + distogram

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        if self.imitate_trunk:
            z = z + self.msa_module(z, s_inputs, feats)

            s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            #! remove the last norm for z
            # s, z = self.final_s_norm(s), self.final_z_norm(z)
            s = self.final_s_norm(s)

        else:
            s_t, z_t = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

            # AF3 has residual connections, we remove them
            s = s_t
            z = z_t

        out_dict = {"s_pred": self.sequence_head(s)}
        return out_dict

    def sequence_loss(self, out, batch, multiplicity=1):
        mask_indices = batch['mask_indices'].repeat_interleave(multiplicity, 0)        # B, l
        mask_target = batch['mask_target'].repeat_interleave(multiplicity, 0)          # B, l
        mask_logits = torch.gather(                 # B, l, N
            out['s_pred'], dim=1,
            index=mask_indices[:, :, None].expand(-1, -1, const.num_tokens)
        )
        pad_mask = mask_target != 0  # in collate, pad value is 0
        
        logits = mask_logits.view(-1, const.num_tokens)
        target = mask_target.view(-1)
        mask = pad_mask.view(-1).float()
        seq_loss = F.cross_entropy(logits, target, reduction='none')
        seq_loss = seq_loss * mask
        seq_loss = torch.sum(seq_loss) / torch.sum(mask)
        
        pred = torch.argmax(logits, dim=-1)
        seq_acc = torch.sum((pred == target) * mask) / torch.sum(mask)

        mask_seq_rate = batch['mask_seq_rate']              # B
        mask_msa_rate = batch['mask_msa_rate']
        avg_seq_mask = torch.mean(mask_seq_rate)
        avg_msa_mask = torch.mean(mask_msa_rate)

        return seq_loss, seq_acc, avg_seq_mask, avg_msa_mask