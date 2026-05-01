import sys
from dataclasses import dataclass
from typing import Optional

import torch
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from boltz.data.module.training import DataConfig
from afgen.training import AFGenTrainingDataModule
from afgen.initializer import instantiate

@dataclass
class TrainConfig:
    name: str
    output: str
    data: DataConfig
    model: LightningModule
    pretrained: Optional[str] = None
    resume: Optional[str] = None
    disable_checkpoint: bool = False
    save_top_k: Optional[int] = 1
    matmul_precision: Optional[str] = None
    load_confidence_from_trunk: Optional[bool] = False
    load_sequence_from_trunk: Optional[bool] = True
    find_unused_parameters: Optional[bool] = False
    validation_only: bool = False
    strict_loading: bool = True
    trainer: Optional[dict] = None


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    raw_config = omegaconf.OmegaConf.load(raw_config)
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    cfg = instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer

    # Create objects
    data_config = DataConfig(**cfg.data)
    data_module = AFGenTrainingDataModule(data_config)
    model_module = cfg.model

    if cfg.pretrained and not cfg.resume:
        checkpoint = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
        # Load the pretrained weights into the confidence module
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("structure_module") and not key.startswith(
                "distogram_module"
            ):
                if cfg.load_confidence_from_trunk:
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
                if cfg.load_sequence_from_trunk:
                    new_key = "sequence_module." + key
                    new_state_dict[new_key] = value
        new_state_dict.update(checkpoint["state_dict"])
        checkpoint["state_dict"] = new_state_dict
        print(f"Loading model from {cfg.pretrained}")
        model_module.load_state_dict(checkpoint['state_dict'], strict=False)

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        monitor = "val/sequence_acc" if cfg.model.sequence_prediction else "val/lddt"
        mc = ModelCheckpoint(
            monitor=monitor,
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="max",
            every_n_epochs=1,
        )
        callbacks = [mc]

    logger = TensorBoardLogger(cfg.output, name=cfg.name)

    # Set up trainer
    strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)
    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2:]
    train(arg1, arg2)