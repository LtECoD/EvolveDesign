import click
import torch
from pathlib import Path
from dataclasses import asdict
from typing import Literal, Optional
from pytorch_lightning import (
    Trainer,
    seed_everything,
)
from pytorch_lightning.strategies import DDPStrategy

from boltz.data.types import Manifest
from boltz.main import (
    process_inputs,
    check_inputs,
    PairformerArgs,
    MSAModuleArgs,
    BoltzSteeringParams,
    BoltzProcessedInput,
    BoltzDiffusionParams
)

from afgen.writer import Writer
from afgen.model import AFGenModel
from afgen.inference import AFGenInferenceDataModule, BatchUpdater


@click.group()
def cli() -> None:
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz.",
    default="~/.boltz",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--predict_structure",
    type=bool,
    is_flag=True,
    help="Whether to predict structure. Default is False.",
)
@click.option(
    "--update_msa",
    type=bool,  
    is_flag=True,
    help="Whether to update msa in the last predict step. Default is False.",
)
@click.option(
    "--temperature",
    type=float,
    help="The temperature to use for prediction. Default is 1.0.",
    default=1.0,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--step_scale",
    type=float,
    help="The step size is related to the temperature at which the diffusion process samples the distribution."
    "The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
    default=1.638,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is False.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=1,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--no_potentials",
    is_flag=True,
    help="Whether to disable potentials. Default is False.",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)
@click.option(
    "--sequence_prediction",
    is_flag=True,
    help="Whether to predict sequence"
)
@click.option(
    "--confidence_prediction",
    is_flag=True,
    help="Whether to predict confidence"
)
@click.option(
    "--not_use_trifast",
    is_flag=True,
    help="Whether to not use the Trifast model. Default is True.",
)
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    predict_structure: bool = False,
    no_potentials: bool = False,
    update_msa: bool = False,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    temperature: float = 1.0,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    sequence_prediction: bool = False,
    confidence_prediction: bool = False,
    not_use_trifast: bool = False,
) -> None:
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    # torch.set_float32_matmul_precision("highest")
    torch.set_float32_matmul_precision("high")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = AFGenInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )
    updater = BatchUpdater(ccd_path=cache / "protein_ccd.pkl")

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "temperature": temperature,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        "predict_structure": predict_structure,
        "update_msa": update_msa,
    }
    pred_writer = Writer(
        out_dir=Path(out_dir)/"predictions",
        output_format=output_format,
    )

    steering_args = BoltzSteeringParams()
    if no_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False
    
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale

    # pairformer_args = PairformerArgs(use_trifast=(accelerator != "cpu"))
    # msa_module_args = MSAModuleArgs(use_trifast=(accelerator != "cpu"))
    pairformer_args = PairformerArgs(use_trifast=not not_use_trifast)
    msa_module_args = MSAModuleArgs(use_trifast=not not_use_trifast)

    model_module: AFGenModel = AFGenModel.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        pairformer_args=asdict(pairformer_args),
        msa_module_args=asdict(msa_module_args),
        steering_args=steering_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        batch_updater=updater,
        writer=pred_writer,
    )
    model_module.eval()

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


if __name__ == "__main__":
    cli()