# Vision-transformers-for-image-classification
This repository contains implementation of vision transformers for image classification


# Overview

This notebook implements Vision Transformer (ViT) model by Alexey Dosovitskiy et al for image classification, and demonstrates it on the Cassava Leaf Disease Classification dataset.

For from scratch implementation of ViT check out this notebook: <br>
https://www.kaggle.com/raufmomin/vision-transformer-vit-from-scratch

Research Paper: https://arxiv.org/pdf/2010.11929.pdf <br>
Github (Official) Link: https://github.com/google-research/vision_transformer <br>
Github (Keras) Link: https://github.com/faustomorales/vit-keras

#### Highlights of this notebook:
1. Pre-trained Vision Transformer (vit_b32) on imagenet21k dataset
2. Label Smoothing of 0.3
3. Custom data augmentation for ImageDataGenerator
4. RectifiedAdam Optimizer


# Model Architecture

![image](https://github.com/SujanSharma07/Vision-transformers-for-image-classification/assets/50839018/47344d8b-8d60-46a5-8a9e-f40b64cbbc06)


# Available models
There are models pre-trained on imagenet21k for the following architectures:
ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32 and ViT-H/14. 
There are also the same models pre-trained on imagenet21k and fine-tuned on imagenet2012.

![image.png](attachment:image.png)


# Experimental Setup
For our experiments, we utilized Kaggle as our computing platform. Kaggle offers robust computational
resources, including access to GPUs and TPUs, which are essential for training deep learning models
efficiently. We uploaded our dataset to Kaggle and leveraged the resources to accelerate our model
training and evaluation processes.

# Tools and Libraries
Python was employed for the implementation, utilizing several libraries for data manipulation,
visualization, and model training. The primary libraries used include:
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
TensorFlow: For building and training the deep learning model.
TensorFlow Addons: For additional functionalities that extend TensorFlow's capabilities.
Matplotlib: For plotting and visualizing data.
Scikit-learn: For evaluating model performance with metrics like confusion matrix and classification
report.
Seaborn: For creating visually appealing statistical graphics.

# Pretrained Vision Transformer Model
A pre-trained Vision Transformer (ViT) model was implemented through the vit_keras package.
For our model, we used a pre-trained Vision Transformer (ViT) model implemented via the vit_keras
package. Superior performance in picture classification tasks has been demonstrated by the ViT model,
which was first presented in the publication "An image is worth 16x16 words: transformers for image
recognition at scale," [9]. Pre-trained weights on the ImageNet21K and ImageNet2012 datasets, which
are saved in npz format, are provided by this package.
Because a pre-trained model has already learned a comprehensive collection of features from extensive
picture datasets, using one drastically cuts down the time and computing power essential for the training
purpose. It also makes it possible to effectively fine-tune the model in the particular dataset.
With the aid of several Python libraries, Kaggle’s processing power, and the sophisticated features of the
Vision Transformer model, we were able to conduct our trials efficiently and provide reliable results for
our picture classification challenges.

# Model summary
![image](https://github.com/SujanSharma07/Vision-transformers-for-image-classification/assets/50839018/46654e2a-714e-47c9-9428-c070170a9fe5)
