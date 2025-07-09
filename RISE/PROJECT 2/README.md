# HANDDIGIT CLASSIFIER
---
The focus on developing a multi-task learning (MTL) model using the Keras framework with TensorFlow as the backend. Multi-task learning is a machine learning paradigm where a single model is trained to perform multiple related tasks simultaneously, leveraging shared representations to improve efficiency and performance. In this project, the model is designed to perform two tasks on a modified version of the MNIST dataset: 

> Digit Classification: Identifying the digit (0–9) in a grayscale MNIST image.

> Color Classification: Determining the dominant color channel (red or green) in a synthetically colored version of the image.

The project uses the MNIST dataset, which consists of 60,000 training and 10,000 testing 28x28 grayscale images of handwritten digits. The images are preprocessed to create RGB images with a randomly assigned red or green color channel, adding a second task to the traditional digit classification problem. The model is built using Keras, trained with a custom data generator, and evaluated on both tasks simultaneously. The project demonstrates key concepts in deep learning, including convolutional neural networks (CNNs), data preprocessing, and model evaluation.

## The key components of the proposed system are:
	
  - Custom Dataset: A modified MNIST dataset where grayscale images are converted to RGB with a randomly assigned red or green color channel.
  - Multi-Task Model: A CNN with shared convolutional layers and two output heads: one for digit classification (softmax over 10 classes) and one for color classification (sigmoid for binary classification).
  - Data Generator: A tf.data.Dataset-based generator to efficiently preprocess and feed data in batches.
  - Training Pipeline: A training process that optimizes both tasks using weighted loss functions (sparse_categorical_crossentropy for digits, binary_crossentropy for color).
  - Evaluation and Visualization: Tools like TensorBoard for monitoring, model architecture plots, and prediction visualizations to assess performance.
