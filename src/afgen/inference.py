import torch
import pickle
import shutil
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from torch import Tensor
from dataclasses import replace
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.types import (
    Manifest,
    Structure,
    Chain,
    Residue,
    Atom,
    Input,
    Record,
    MSA,
    Tokenized
)
from boltz.data.module.inference import (
    pad_to_max,
    load_input,
    PredictionDataset,
)
from boltz.main import compute_msa
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.schema import get_conformer
from boltz.data.feature.featurizer import (
    process_msa_features,
    process_atom_features
)

from afgen.featurizer import AFGenFeaturizer
from afgen.tokenizer import AFGenTokenizer, convert_atom_name


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "structure",
            "tokenized",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class AFGenPredictionDataset(PredictionDataset):
    def __getitem__(self, idx: int) -> dict:
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, self.target_dir, self.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
                compute_constraint_features=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        features["record"] = record
        features['structure'] = input_data.structure
        features['tokenized'] = tokenized
        return features


class AFGenInferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir

    def predict_dataloader(self) -> DataLoader:
        dataset = AFGenPredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
        )
        dataset.tokenizer = AFGenTokenizer()
        dataset.featurizer = AFGenFeaturizer(check_mask=False)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "structure",
                "tokenized",
            ]:
                batch[key] = batch[key].to(device)
        return batch


class BatchUpdater:
    def __init__(self, ccd_path: Path):
        self.tokenizer = AFGenTokenizer(mask_msa=False)
        self.components = pickle.load(ccd_path.open("rb"))

    def update_batch(self, batch, target: str, res_type=None, unmask_mask=None, **kwargs):
        """ Update the batch with the residue type.
        Args:
            batch (dict): feats, record and structure
            res_type (torch.LongTensor): 1, N
            target (str): to update residue, atom, or msa
            unmask_mask (torch.BoolTensor): 1, N, optional, indicate residue unmasked
            msa_dir (Path): msa directory, optional
        """
        #todo, add batch support

        structure: Structure = batch['structure'][0]
        record: Record = batch['record'][0]
        tokenized: Tokenized = batch['tokenized'][0]

        if target == "residue":
            residues: np.ndarray = structure.residues
            batch['res_type'] = torch.nn.functional.one_hot(
                res_type, num_classes=const.num_tokens).float()

            # only update protein sequence            
            res_type = res_type.squeeze(0).cpu().numpy()
            batch_prot_mask = batch['mol_type'].squeeze(0).cpu().numpy() == const.chain_type_ids["PROTEIN"]

            struc_prot_mask = np.array([0] * len(residues), dtype=bool)
            for cid, chain in enumerate(structure.chains):
                if const.chain_types[chain['mol_type']] != "PROTEIN":
                    continue
                rstart, rnum = int(chain['res_idx']), int(chain['res_num'])
                struc_prot_mask[rstart: rstart+rnum] = True
            
            residues['res_type'][struc_prot_mask] = res_type[batch_prot_mask]
            residues['name'][struc_prot_mask] = np.array([const.tokens[i] for i in res_type[batch_prot_mask]])

            structure = replace(structure, residues=residues)
            batch['structure'] = [structure]
        
        elif target == "atom":
            atoms: np.ndarray = structure.atoms
            chains: np.ndarray = structure.chains
            residues: np.ndarray = structure.residues
            unmask_mask = unmask_mask.squeeze(0).cpu().numpy()
            res_type = res_type.squeeze(0).cpu().numpy()

            # update structure
            new_atoms = []
            new_residus = []
            new_chains = []
            cur_atom_idx = 0
            for cid, chain in enumerate(chains):
                chain_atom_idx, chain_atom_num = cur_atom_idx, 0

                chain = chains[cid]
                rstart, rnum = int(chain['res_idx']), int(chain['res_num'])
                chain_residues = residues[rstart: rstart+rnum]

                for rid in range(rnum):
                    residue = chain_residues[rid]
                    astart, anum = int(residue['atom_idx']), int(residue['atom_num'])
                    residue_atoms = atoms[astart: astart+anum]

                    if const.chain_types[chain['mol_type']] != "PROTEIN" or not unmask_mask[rid+rstart]:
                        new_atoms.extend(residue_atoms)
                    else:
                        assert anum == 5, "unmaked token is not UNK token"
                        res_name = const.tokens[res_type[rid+rstart]]
                        ref_mol = self.components[res_name]                        
                        ref_conformer = get_conformer(ref_mol)
                        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
                        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_name]]
                        anum = len(ref_atoms)

                        for ref_atom in ref_atoms:
                            # Get atom name
                            atom_name = ref_atom.GetProp("name")
                            # Get conformer coordinates
                            ref_coords = ref_conformer.GetAtomPosition(ref_atom.GetIdx())
                            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)

                            # Add atom to list
                            new_atoms.append(np.array((
                                convert_atom_name(atom_name),
                                ref_atom.GetAtomicNum(),
                                ref_atom.GetFormalCharge(),
                                (0, 0, 0),
                                ref_coords,
                                True,
                                const.chirality_type_ids.get(
                                    ref_atom.GetChiralTag(), const.chirality_type_ids[const.unk_chirality_type]
                                ),
                            ), dtype=Atom))

                    
                    # determine center and disto index
                    if const.chain_types[chain['mol_type']] != "PROTEIN":
                        _center = residue['atom_center'] - astart
                        _disto = residue['atom_disto'] - astart
                    else:
                        _center = const.res_to_center_atom_id[residue['name']]
                        _disto = const.res_to_disto_atom_id[residue['name']]

                    new_residus.append(np.array((
                        residue['name'],
                        residue['res_type'],
                        residue['res_idx'],
                        cur_atom_idx,
                        anum,
                        _center+cur_atom_idx,
                        _disto+cur_atom_idx,
                        residue['is_standard'],
                        residue['is_present']
                    ), dtype=Residue))
                    cur_atom_idx += anum
                    chain_atom_num += anum

                new_chains.append(np.array((
                    chain['name'],
                    chain['mol_type'],
                    chain['entity_id'],
                    chain['sym_id'],
                    chain['asym_id'],
                    chain_atom_idx,
                    chain_atom_num,
                    chain['res_idx'],
                    chain['res_num'],
                    chain['cyclic_period']
                ), dtype=Chain))
            structure = replace(
                structure,
                atoms=np.array(new_atoms, dtype=Atom),
                residues=np.array(new_residus, dtype=Residue),
                chains=np.array(new_chains, dtype=Chain), 
            )

            # update atom features
            data = Input(structure=structure, msa=tokenized.msa)
            tokenized = self.tokenizer.tokenize(data)
            atom_features = process_atom_features(
                tokenized,
                atoms_per_window_queries=32,
                min_dist = 2.0,
                max_dist = 22.0,
                num_bins = 64,
                max_tokens = None,
                max_atoms = None,
            )
            atom_features = collate([atom_features])
            atom_features = {k: v.to(batch['res_type'].device) for k, v in atom_features.items()}

            # update msa features
            msa_features = process_msa_features(
                tokenized,
                max_seqs_batch=const.max_msa_seqs,
                max_seqs=const.max_msa_seqs
            )
            msa_features = collate([msa_features])
            msa_features = {k: v.to(batch['res_type'].device) for k, v in msa_features.items()}

            batch.update(atom_features)
            batch.update(msa_features)

            batch['structure'] = [structure]

        elif target == "msa":
            sequences = {}
            for chain in structure.chains:
                if chain['mol_type'] == const.chain_type_ids["PROTEIN"]:
                    entity_id = chain['entity_id']
                    cres = structure.residues[chain['res_idx']:chain['res_idx']+chain['res_num']]
                    seq = "".join([const.prot_token_to_letter[const.tokens[i]] for i in cres['res_type']])
                    sequences[entity_id] = seq

            to_generate = {}
            new_msa_maps = {}
            target_id = record.id
            for chain in record.chains:
                # Add to generate list, assigning entity id
                if chain.mol_type == const.chain_type_ids["PROTEIN"]:
                    entity_id = chain.entity_id
                    msa_id = f"{target_id}_{entity_id}"
                    to_generate[msa_id] = sequences[entity_id]
                    chain.msa_id = kwargs['msa_dir'] / f"{msa_id}.csv"
                    new_msa_maps[chain.chain_id] = chain.msa_id

            if kwargs['msa_dir'].exists():
                shutil.rmtree(kwargs['msa_dir'])
            kwargs['msa_dir'].mkdir(parents=True, exist_ok=True)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=kwargs['msa_dir'],
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
            )

            msas = {}
            for chain_id, msa_id in new_msa_maps.items():
                msa_path = Path(msa_id)
                msa: MSA = parse_csv(msa_path, max_seqs=2048)
                msas[chain_id] = msa
            tokenized = replace(tokenized, msa=msas)

            msa_features = process_msa_features(
                data=tokenized,
                max_seqs_batch=const.max_msa_seqs,
                max_seqs=const.max_msa_seqs
            )
            msa_features = collate([msa_features])
            msa_features = {k: v.to(batch['res_type'].device) for k, v in msa_features.items()}
            batch.update(msa_features)

        else:
            raise ValueError(f"target {target} is not supported")
        return batch
