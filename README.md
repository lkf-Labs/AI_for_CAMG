# AI_for_CAMG

This code is associated with the paper **"Open Benchmark Dataset for AI-Based Quantitative Analysis of Meibomian Gland in Children and Adolescents"** by Li Li et al.

## Introduction
 ![overview1](https://github.com/user-attachments/assets/87d5c84a-32c0-41c9-8de1-61220a517e0a)

 <div align="center">
   <img width="446" alt="overview" src="[https://github.com/user-attachments/assets/a98daeef-5fc5-4b37-9cb5-a111d4d78cb4](https://github.com/user-attachments/assets/87d5c84a-32c0-41c9-8de1-61220a517e0a)">
   <img width="446" alt="overview" src="https://github.com/user-attachments/assets/a98daeef-5fc5-4b37-9cb5-a111d4d78cb4">
</div>



**Abstract**  Meibomian glands play a crucial role in maintaining tear film stability by secreting the lipid layer, and dysfunction of these glands is a major contributor to tear film instability and exacerbation of dry eye symptoms. Current evaluations heavily depend on clinicians' subjective judgment and experience, resulting in significant variability in diagnoses. The development of artificial intelligent (AI) algorithms relies on high-quality open datasets. To bridge the gap, we present a new open-access dataset named the Children and Adolescents Meibomian Gland (CAMG) dataset. This dataset includes 1,114 infrared images of upper eyelid meibomian glands from 600 participants aged 4 to 18 years, collected between June 2020 and July 2024. The images underwent preprocessing, including denoising and normalization, and were manually annotated by three junior ophthalmologists and one senior ophthalmologist. Subsequently, AI algorithms were used for segmentation, and the results were reviewed and corrected by an experienced ophthalmologist. The dataset also includes demographic data and various quantitative parameters of the meibomian glands, such as average gland count, length, width, area, and the characteristics of the central five glands. In addition, we provided meibomian gland parameters for children and adolescents of different ages and genders, offering a reference for clinical practice.




## Setup
Install the following packages: torch torchvision kornia opencv-python matplotlib PyYAML scikit-learn Pillow
Alternatively, create a virtual environments and install the packages of `requirements.txt` using the following command:
```bash
pip3 install -r requirements.txt
```

## Training And Evaluation
Before running the experiment, you need to configure some relevant hyperparameters in the `train_config.yaml` file.
We train the U-Net model and evaluate its performance on the Meibomian Gland dataset using the following command:
```bash
python3 main.py
```
## Declaration
This source code provides only some core functionalities. For more details, please contact `231027062@fzu.com`.
