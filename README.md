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

## Input Format Examples

### Monomer (no MSA)

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG
      msa: empty
```

### Multimer

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA
```

### With Ligand

```yaml
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
      msa: ./examples/msa/seq1.a3m
  - ligand:
      id: [C, D]
      ccd: SAH
  - ligand:
      id: [E, F]
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
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

说明：

- 该配置为轻量测试配置，便于检查训练流程是否可跑通。
- 训练输出默认写入 output/sequence。

### 2) 多卡/多机训练（SLURM）

仓库已提供示例脚本：

```bash
bash scripts/train/train_sequence.sh
```

或（续训版本）：

```bash
bash scripts/train/train_sequence_th_resume.sh
```

注意：

- 脚本依赖 SLURM 变量（如 SLURM_NNODES、SLURM_JOB_NODELIST）。
- 请先按集群实际资源修改分区、节点数、GPU 数、配置文件路径。

### 3) 推理预测

基础用法：

```bash
python src/afgen/main.py predict <input_yaml> --out_dir <output_dir>
```

示例（使用仓库测试样例）：

```bash
python src/afgen/main.py \
  predict \
  test/example/multimer_unk.yaml \
  --out_dir test/output/multimer_unk \
  --checkpoint output/sequence/train_full_msa/version_0/checkpoints/epoch=0_w_conf.ckpt \
  --seed 42 \
  --diffusion_samples 2 \
  --sampling_steps 200 \
  --write_full_pae \
  --write_full_pde \
  --output_format pdb \
  --use_msa_server \
  --override
```

常用参数：

- --checkpoint: 指定模型权重；不指定时会尝试使用缓存内默认权重
- --predict_structure: 启用结构预测
- --update_msa: 在最后预测阶段更新 MSA
- --recycling_steps: recycling 次数
- --sampling_steps: 采样步数
- --diffusion_samples: diffusion 样本数
- --temperature: 采样温度
- --output_format: pdb 或 mmcif
- --use_msa_server / --msa_server_url: 使用远程 MSA 服务

提示：test/test_predict.sh 中出现的 --sequence_steps 参数在当前 CLI 中不存在，建议不要使用该参数。

## 输入数据格式（推理）

以 test/example/multimer_unk.yaml 为例：

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
      msa: ./test/example/msa/multimer_a.csv
  - protein:
      id: B
      sequence: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      msa: ./test/example/msa/multimer_b.csv
```

## 训练配置说明

主要训练配置在 scripts/train/configs。

建议从以下文件开始：

- scripts/train/configs/sequence_test1.yaml: 最小可运行测试配置
- scripts/train/configs/sequence.yaml: 常用序列训练配置
- scripts/train/configs/sequence_th_resume.yaml: 续训配置示例

关键字段：

- data.datasets[].target_dir / msa_dir: 训练数据路径
- data.symmetries: symmetry 文件路径
- model.sequence_prediction / confidence_prediction: 任务开关
- model.training_args: 学习率、采样步数、损失权重等
- trainer.devices / num_nodes / precision: 分布式与硬件配置

## 评估

结构评估脚本：

```bash
python scripts/eval/run_evals.py <pred_dir> <ref_dir> <out_dir> --format boltz --testset test --mount <挂载根目录>
```

该脚本通过 Docker 调用 OpenStructure 镜像（openstructure-0.2.8），请确保：

- 本机可执行 docker
- 对应路径已正确挂载
- 有权限执行 sudo docker run

## 常见问题

1. ModuleNotFoundError: afgen / boltz

- 未设置 PYTHONPATH，执行：
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

2. 推理参数报错 unknown option

- 先运行 python src/afgen/main.py predict --help 检查当前可用参数。
- 注意旧脚本中的部分参数可能已过期（如 --sequence_steps）。

3. 多机训练 rendezvous 失败

- 检查 MASTER_ADDR、MASTER_PORT、节点间网络连通。
- 确保 torchrun 的 nnodes、nproc_per_node 与实际 SLURM 资源一致。

## 许可

本项目使用仓库根目录的 LICENSE。