# AI_for_CAMG

This code is associated with the paper **"Open Benchmark Dataset for AI-Based Quantitative Analysis of Meibomian Gland in Children and Adolescents"** by Li et al.

## Setup
Install the following packages: `torch torchvision kornia opencv-python matplotlib PyYAML scikit-learn Pillow`
Alternatively, create a virtual environments and install the packages of `requirements.txt` using the following command:
```bash
pip3 install -r requirements.txt
```

## Training and Evaluation
Before running the experiment, you need to configure some relevant hyperparameters in the `train_config.yaml` file.
We train the U-Net model and evaluate its performance on the Meibomian Gland dataset using the following command:
```bash
python3 src/main.py
```
## Declaration
This source code provides only some core functionalities. For more details, please contact `231027062@fzu.com`.
