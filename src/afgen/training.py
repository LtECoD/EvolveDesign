import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data.sample.sampler import Sample
from boltz.data.types import Manifest, Record, Input, Structure, MSA, Connection
from boltz.data.module.training import (
    get_symmetries,
    collate,
    DataConfig,
    TrainingDataset,
    ValidationDataset,
)

from afgen.cropper import AFGenCropper
from afgen.featurizer import AFGenFeaturizer
from afgen.tokenizer import AFGenTokenizer
from afgen.cluster  import ClusterSampler


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: ClusterSampler
    cropper: AFGenCropper
    filters: Optional[list] = None
    split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    msa_dir: Path
    manifest: Manifest
    prob: float
    sampler: ClusterSampler
    cropper: AFGenCropper
    tokenizer: AFGenTokenizer
    featurizer: AFGenFeaturizer


def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    # Load the structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1 and msa_id != "":
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas, record)


class AFGenTrainingDataset(TrainingDataset):
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """

        # Pick a random dataset
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.probs,
        )
        dataset: Dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])

        # Get the structure
        try:
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return self.__getitem__(idx)

        # Compute crop
        try:
            if self.max_tokens is not None:
                crop_mask = dataset.cropper.crop(
                    input_data.structure,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(
                input_data, crop_mask, max_tokens=self.max_tokens, max_atoms=self.max_atoms
            )
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            features = dataset.featurizer.process(
                tokenized,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
                compute_constraint_features=False,
            )
        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        return features


class AFGenValidationDataset(ValidationDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probs = [d.prob for d in kwargs['datasets']]
        self.samples = []
        for dataset in kwargs['datasets']:
            records = dataset.manifest.records
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # # Pick dataset based on idx
        # for dataset in self.datasets:
        #     size = len(dataset.manifest.records)
        #     if self.overfit is not None:
        #         size = min(size, self.overfit)
        #     if idx < size:
        #         break
        #     idx -= size

        # # Get a sample from the dataset
        # record = dataset.manifest.records[idx]

        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.probs,
        )
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])
        record = sample.record

        # Get the structure
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                crop_mask = dataset.cropper.crop(
                    input_data.structure,
                    max_tokens=self.max_tokens,
                    random=self.random,
                    max_atoms=self.max_atoms,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(
                input_data, crop_mask, max_tokens=self.max_tokens, max_atoms=self.max_atoms
            )
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
                compute_constraint_features=False,
            )
        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        return features


class AFGenTrainingDataModule(pl.LightningDataModule):
    """DataModule for boltz."""

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

        # Load symmetries
        symmetries = get_symmetries(cfg.symmetries)

        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        for data_config in cfg.datasets:
            # Set target_dir
            target_dir = Path(data_config.target_dir)
            msa_dir = Path(data_config.msa_dir)

            # Load manifest
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Split records if given
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {x.lower() for x in f.read().splitlines()}

                train_records = []
                val_records = []
                for record in manifest.records:
                    if record.id.lower() in split:
                        val_records.append(record)
                    else:
                        train_records.append(record)
            else:
                train_records = manifest.records
                val_records = []

            # Filter training records
            train_records = [
                record
                for record in train_records
                if all(f.filter(record) for f in cfg.filters)
            ]
            # Filter training records
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    target_dir,
                    msa_dir,
                    train_manifest,
                    data_config.prob,
                    data_config.sampler,
                    data_config.cropper,
                    cfg.tokenizer,
                    cfg.featurizer,
                )
            )

            # Create validation dataset
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset(
                        target_dir,
                        msa_dir,
                        val_manifest,
                        data_config.prob,
                        data_config.sampler,
                        data_config.cropper,
                        cfg.tokenizer,
                        cfg.featurizer,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.manifest.records)}")

        for dataset in val:
            dataset: Dataset
            print(f"Validation dataset size: {len(dataset.manifest.records)}")

        # Create wrapper datasets
        self._train_set = AFGenTrainingDataset(
            datasets=train,
            samples_per_epoch=cfg.samples_per_epoch,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
        )
        self._val_set = AFGenValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            crop_validation=cfg.crop_validation,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )
