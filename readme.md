# **VQ-VAE with PixelCNN for MNIST Image Generation** 🎨

This repository presents an implementation of the **Vector Quantized Variational Autoencoder (VQ-VAE)** integrated with **PixelCNN** for generating images from the **MNIST** dataset. The model effectively encodes input images into a discrete latent space, enabling the generation of new samples that closely resemble handwritten digits.

---

## **Table of Contents** 📑
- [🛠️Requirements](#requirements)
- [📚Dataset](#dataset)
- [🔍Model Architecture](#model-architecture)
- [📊 Results](#results)
- [💻 Usage](#usage)

---

## 🛠️ Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- TensorFlow Probability

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib tensorflow-probability
```

---

## 📚 Dataset

The model is trained on the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). The dataset is divided into 60,000 training images and 10,000 test images.

---

## 🔍 Model Architecture

The implementation consists of the following main components:
- **Encoder:** A series of convolutional layers that compress the input images into a lower-dimensional latent representation.
- **Vector Quantizer:** A layer that maps continuous latent representations to discrete embeddings. It includes a commitment loss to ensure the encoder learns to commit to specific embeddings.
- **Decoder:** A series of transposed convolutional layers that reconstruct images from the quantized embeddings.
- **PixelCNN:** A conditional generative model that generates images pixel-by-pixel based on the encoded indices.

---

## 📊 Results

The trained model can generate new samples that resemble the MNIST digits. After training, the script visualizes the original images and their reconstructions, as well as the learned discrete embeddings.

---

## 💻 Usage

To generate new images after training, you can modify the training script or create a separate script that utilizes the trained PixelCNN and VQ-VAE models to sample new digits.
