pip install pytorch-lightning hydra-core omegaconf click tqdm

# AFGen

AFGen is a protein structure and sequence generation toolkit based on the Boltz framework, supporting training, inference, and evaluation for structure, confidence, and sequence prediction tasks.

## Features

- **Training**: Structure, confidence, and sequence prediction (mainly sequence)
- **Inference**: Structure and sequence prediction from YAML input
- **Evaluation**: lDDT, TM-score, DockQ, etc. via OpenStructure (Docker)

Core code is in `src/afgen`, reusing some components from `src/boltz`.

## Directory Overview

- `src/afgen`: AFGen core (model, data modules, inference, CLI)
- `src/boltz`: Boltz framework code
- `scripts/train`: Training entry points, configs, cluster scripts
- `scripts/eval`: Evaluation scripts
- `test`: Minimal test configs and input examples
- `output`: Default output directory
- `log`: Log files

## Environment Setup

No fixed requirements file is provided. Minimal dependencies (install according to your CUDA version):

- Python 3.10+
- PyTorch
- PyTorch Lightning
- Hydra / OmegaConf
- Click
- tqdm

Example (using Conda):

```bash
conda create -n afgen python=3.11 -y
conda activate afgen
# Install torch according to your CUDA version, e.g.:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning hydra-core omegaconf click tqdm
```

Set Python path:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Quick Start

### 1) Minimal Single-GPU Training (test config)

```bash
torchrun \
  --nproc_per_node 1 \
  scripts/train/AFGen_train.py \
  scripts/train/configs/sequence_test1.yaml
```

Edit the config YAML to set your data paths and output directory.

### 2) Inference (Structure/Sequence Prediction)

Prepare a YAML input (see examples below), then run:

```bash
python -m afgen.main predict \
  <input.yaml> \
  --out_dir <output_dir> \
  --checkpoint <model.ckpt> \
  --devices 1 \
  --accelerator gpu \
  --predict_structure \
  --sequence_prediction
```

Key options:
- `--predict_structure`: Enable structure prediction
- `--sequence_prediction`: Enable sequence prediction
- `--confidence_prediction`: Enable confidence prediction
- `--recycling_steps`, `--sampling_steps`, `--diffusion_samples`, etc. for advanced control

### 3) Evaluation

Evaluation scripts are in `scripts/eval/`. For structure metrics (lDDT, TM-score, DockQ), OpenStructure (Docker) is required. Example usage:

```bash
python scripts/eval/run_evals.py --help
```

## Configuration

Edit YAML configs in `scripts/train/configs/` for training, and use the provided YAML input templates in `test/example/` for inference.

## Notes

- For multi-GPU or cluster training, see `scripts/train/train_sequence.sh` for SLURM usage.
- For custom datasets, adjust the `data:` section in the config YAML.
- Some features require external tools (e.g., OpenStructure for evaluation).

## License

See [LICENSE](LICENSE).
  test/configs/sequence_test1.yaml
```
