# VAE-GAN-Pokémon-generator
Variational Autoencoder trained on 809 Pokémon sprites for image generation, reconstruction, and latent space interpolation. Includes β-VAE, augmentations, and PyTorch training pipeline.

## Overview
This project focuses on generating Pokémon images using a **Variational Autoencoder (VAE)**, with a smaller experimental **GAN** side project.  
The VAE allows image reconstruction, latent space interpolation, and new Pokémon image generation. The GAN experiments explore higher-quality sample generation.


Because the dataset is much smaller than typical image-generation datasets, this project focuses on:
- training stability  
- augmentation strategies  
- β-VAE experiments  
- latent space interpretability  

This repository demonstrates practical skills in:
- Deep Learning / VAE architectures  
- PyTorch training pipelines  
- Image preprocessing & augmentation  
- Generative model evaluation

---

## Features
-  VAE architecture with configurable latent dimension  
-  β-VAE option for disentanglement  
-  Reconstruction of Pokémon images  
-  Latent space interpolation  
-  Augmentation to improve generalization on small dataset  
-  Checkpoint saving and reproducible experiments
-  Experimental GAN models for additional generative quality
-  Lightweight training 

---
## Repository Structure

pokemon-generator/
├── Vae/
│ ├── keep/ # Output images (reconstructions, generated, interpolation)
│ ├── generate.py
│ ├── interpolate.py
│ ├── model.py
│ ├── preprocess.py
│ └── train_vae.py
│
├── Gans/
│ ├── model1.py
│ ├── model2.py
│ ├── model3.py # Best GAN version
│ ├── training1.py
│ ├── training2.py
│ ├── training3.py # Best training version
│ ├── data_loader1.py
│ └── data_loader2.py
│
├── README.md
└── requirements.txt

---

## Dataset
**Pokémon Image Dataset (809 sprites)**  

The Pokémon image dataset used for training was **provided directly by my professor** as part of the course materials.  
The **original source or license information is unknown**.  

Because of this, the dataset is **not included** in this repository.  
To run the project, you will need to **supply your own Pokémon image dataset** (e.g., from Kaggle or your own collection) and place it in: "./images"

---

## Outputs

VAE-generated samples, reconstructions, and interpolation experiments are saved in: Vae/keep/

GAN outputs are saved according to the respective training scripts.

---

## Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy / Pandas  
- Matplotlib  
- TQDM  

---

## Notes

- Focus on **VAE files** for primary experiments.  
- Use **highest-numbered GAN files** (`model3.py`, `training3.py`, `data_loader2.py`) if you explore GAN experiments.  
- This repository showcases deep learning skills in generative modeling and PyTorch.


**Note:**  
All generated reconstructions, samples, and interpolation results are saved in: Vae/keep/
