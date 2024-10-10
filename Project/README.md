# GAN-BERT: Deep Learning Course Project

This repository contains the implementation of the GAN-BERT project, which was part of the Deep Learning course at Sharif University of Technology under the supervision of Dr. Fatemizadeh. The project aims to enhance text classification tasks using a combination of Generative Adversarial Networks (GANs) and BERT, focusing on semi-supervised learning methods to improve performance with limited labeled data.

## Project Overview

### Goal
The goal of this project was to implement a novel text classification model called **GAN-BERT**, which leverages the advantages of both GANs and BERT. The project primarily focuses on using limited labeled data along with abundant unlabeled data, thereby implementing a semi-supervised learning approach.

### Key Concepts
1. **SS-GAN**: The **Semi-Supervised GAN** model was explored as the base architecture, involving a generator and a discriminator network. The discriminator not only distinguishes between real and fake samples but also classifies real samples into specific classes.

2. **GAN-BERT**: A generalization of the BERT model used in conjunction with a GAN-like structure. The generator produces samples to match the features of real text data, while the discriminator classifies the input as real or fake and assigns a specific class to real samples.

3. **Dataset**: The dataset used included text samples from multiple sources, including Wikipedia, Reddit, Arxiv, and others. The dataset was comprised of texts produced by various language models (e.g., ChatGPT, Bloomz, Davinci).

## Methodology

### Steps Involved
1. **Data Preprocessing**: The data was formatted and split into training, validation, and test sets. Preprocessing was performed using the BERT tokenizer.

2. **Model Training**: The GAN-BERT model was trained using a combination of labeled and pseudo-labeled data. The training process involved several components:
   - **Generator**: Used to create fake data samples that resemble real data distributions.
   - **Discriminator**: Trained to classify data as either real or generated, and further assign it to one of several classes.

3. **Self-Training**: To improve classification accuracy, the model was first trained on labeled data, and then the trained model was used to create pseudo-labels for the unlabeled data. The model was then retrained on a combination of labeled and pseudo-labeled data.

4. **Evaluation**: Metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were used to evaluate the performance of the model. Experiments with different values of hyperparameters like `max_steps` and the percentage of labeled data were conducted to assess their impact on performance.

### Challenges and Solutions
- **Limited Labeled Data**: The project emphasized semi-supervised learning, utilizing self-training to generate pseudo-labels for unlabeled data, which was challenging but effective.
- **Training Complexity**: The addition of GAN components increased the complexity of training, requiring careful tuning of hyperparameters to balance the generator and discriminator.
- **Adapter Module for Faster Learning**: An adapter layer was added to the BERT model to reduce trainable parameters and speed up the training process while retaining accuracy.

## Results
- The **GAN-BERT** model showed improved performance in text classification tasks, especially when only a small portion of the data was labeled. Using self-training, the model achieved an accuracy of around 40% with 80% of the data being unlabeled.
- The use of **adapter layers** reduced training time by approximately 10%, although with a minor drop in accuracy compared to training without adapters.
- The **G2 generator** architecture provided more realistic fake samples, improving the discriminator's ability to classify samples accurately.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pouria-D/Deep-Learning-Course.git
   cd Deep-Learning-Course/Project
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ and the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Training**:
   Use the provided Jupyter notebook or Python scripts to train the model:
   ```bash
   python train_gan_bert.py
   ```

4. **Evaluation**:
   Run evaluation scripts to check the model's performance on the test set:
   ```bash
   python evaluate.py
   ```

## Folder Structure
- **/data**: Contains training, validation, and test datasets.
- **/models**: Stores pre-trained models and weights for GAN-BERT.
- **/notebooks**: Jupyter notebooks used for exploratory data analysis and training.
- **train_gan_bert.py**: Script to train the GAN-BERT model.
- **evaluate.py**: Script to evaluate model performance.

## Contact
For any further details or questions, feel free to contact me at: [pouria.dadkhah@gmail.com](mailto:pouria.dadkhah@gmail.com).

## Acknowledgments
This project was carried out as part of the Deep Learning course at Sharif University of Technology, under the supervision of Dr. Fatemizadeh. Special thanks to my colleagues Armin Ghoujezadeh and Atieh Mirzaie for their collaboration.

Feel free to explore, fork, and contribute!

