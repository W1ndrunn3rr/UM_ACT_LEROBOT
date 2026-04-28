# UM ACT LeRobot

Training [ACT](https://github.com/huggingface/lerobot) policies on a pick-and-lift task.

## Experiments

| Name | Backbone | VAE | Canny |
|------|----------|-----|-------|
| `baseline` | ResNet18 | yes | no |
| `resnet34_scratch` | ResNet34 | yes | no |
| `resnet34_pretrained` | ResNet34 | yes | no |
| `canny` | ResNet18 | yes | yes |
| `no_vae` | ResNet18 | no | no |

## Setup

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in the values:

```
DATASET_ID=<hf-dataset-repo-id>        # e.g. username/dataset-name
POLICY_REPO_ID=<hf-model-repo-prefix>  # e.g. username/act — experiment name appended automatically
HF_TOKEN=hf_...                        # Hugging Face write token
```

Log in to Weights & Biases:

```bash
wandb login
```

## Running training

**Single experiment:**

```bash
"$VIRTUAL_ENV/bin/python" train.py <experiment_name>
# e.g.
"$VIRTUAL_ENV/bin/python" train.py baseline
"$VIRTUAL_ENV/bin/python" train.py resnet34_scratch
"$VIRTUAL_ENV/bin/python" train.py resnet34_pretrained
```

**Supported ResNet experiments sequentially:**

```bash
chmod +x train.bash
./train.bash
```

Logs are saved to `logs/`, checkpoints to `outputs/train/<experiment_name>/step_XXXXXXX/`, and the final model is pushed to your Hugging Face Hub repo.

**Multi-GPU** (via Accelerate):

```bash
accelerate launch train.py <experiment_name>
```
