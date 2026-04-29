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

## Running

```bash
make train EXP=baseline
make train-all
```

Eval with filtered camera input:

```bash
make eval MODEL=W1ndrunn3rr/act_pick_and_lift_v2_canny FILTER=canny ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_robot
```
