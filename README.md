# Understanding-Disentanglement-in-VAEs
Exploring disentangled latent representations in Variational Autoencoders using the dSprites dataset
# 🧠 Disentanglement in Variational Autoencoders (VAE & β-VAE)

This repository contains the implementation and analysis of **Variational Autoencoders (VAE)** and their extension **β-VAE**, focusing on disentangled representation learning using the **dSprites** dataset.  
All experiments, code, and results correspond to *Homework 1 of Deep Generative Models (DGM)*.

---

## 📘 Overview

The project investigates how the β parameter in the VAE objective influences **disentanglement** of the latent space.  
We train several β-VAE models, compare their reconstruction quality and KL divergence, and evaluate disentanglement using the **Mutual Information Gap (MIG)** metric.

Main objectives:
- Implement and train **β-VAE** with different β values (1, 3, 10).  
- Measure **MIG** to quantify disentanglement.  
- Visualize **latent traversals** to interpret learned factors of variation.

---

## 🧩 Dataset

**dSprites (Matthey et al., DeepMind 2017)**  
- 2D shapes procedurally generated from six independent ground-truth factors.  
- Each image: `64×64` grayscale binary image.  
- Latent variables:  
  - Shape ∈ {square, ellipse, heart}  
  - Scale ∈ [0.5, 1]  
  - Orientation ∈ [0, 2π]  
  - posX, posY ∈ [0, 1]  

---

## ⚙️ Model Architecture

| Encoder | Decoder |
|----------|----------|
| Conv2d(1, 32) → ReLU | Linear(h_dim, 8192) |
| Conv2d(32, 64) → ReLU | Conv2dT(128, 64) |
| Conv2d(64, 128) → ReLU | Conv2dT(64, 32) |
| Flatten → Linear(8192, h_dim) | Conv2dT(32, 1) → Sigmoid |

- Latent dimension `h_dim = 128`  
- Optimizer : Adam (lr = 0.001)  
- Training : 50 epochs on dSprites  

---

## 🧮 Training Objective

\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x)\,\|\,p(z))
\]

β controls the trade-off between reconstruction accuracy and latent factor independence.  
- **β = 1** → Standard VAE  
- **β > 1** → Encourages disentanglement through stronger regularization  

---

## 📊 Results

| β | Description | MIG |
|---|--------------|-----|
| 1 | Standard VAE | 0.0108 |
| 3 | Moderate disentanglement | **0.0198** |
| 10 | Over-regularized | 0.0138 |

### Factor-wise MIG

| Factor | β = 1 | β = 3 | β = 10 |
|---------|--------|--------|---------|
| shape | 0.000 | 0.000 | 0.000 |
| scale | 0.000 | 0.000 | 0.000 |
| orientation | 0.0013 | 0.0010 | 0.0000 |
| posX | 0.0195 | **0.0502** | 0.0492 |
| posY | 0.0331 | **0.0480** | 0.0196 |

---

## 🎨 Latent Traversal

Latent traversals show how changing a single latent dimension affects generated images:

```python
latent_traversal(model_b10, device, dim=0)
