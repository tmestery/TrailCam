# TrailCam: Wildlife Species Recognition from Camera Trap Imagery

 **Project Midterm Report**

 **By:** Tyler Mestery

## Abstract

Camera traps are widely used to monitor wildlife populations, but the large number of images they collect creates a significant burden for researchers who must manually identify species. This project investigates whether lightweight machine learning models can automatically classify wildlife species from camera-trap imagery while remaining practical for deployment in remote environments. The project uses the ENA24 (Eastern North America Camera Trap Detection Dataset), which contains thousands of annotated camera-trap images covering multiple wildlife species. Animal bounding boxes are converted into individual image crops for classification. Four model families will be evaluated: a perceptron, logistic regression, a support vector machine, and a convolutional neural network. A pretrained multimodal vision-language model will also be evaluated using zero-shot classification. The models will be compared using accuracy and macro-F1, with particular attention to whether a locally deployable model can achieve performance close to a larger vision-language model without requiring internet connectivity or per-image API costs.

## 1 Introduction

Camera traps are motion-triggered cameras used to monitor wildlife populations in remote habitats. They can collect thousands of images, but manually identifying animals is time-consuming and creates a backlog for researchers. Machine learning can automate species identification and allow camera-trap data to be processed more efficiently for conservation applications.

This project uses the ENA24 (Eastern North America Camera Trap Detection Dataset), which contains 8,789 camera-trap images and 9,772 bounding-box annotations covering 23 wildlife species. Each annotated animal will be cropped from its original image and used as a labeled example for species classification.

Four machine learning approaches will be compared: a perceptron, logistic regression, a support vector machine (SVM), and a convolutional neural network (CNN). The classical models will use engineered color and texture features, while the CNN will learn features directly from the images. A pretrained multimodal vision-language model will also be evaluated using zero-shot classification.

The main goal is to compare model accuracy and macro-F1 while considering practical deployment requirements. In particular, the project will investigate whether a lightweight model that can run locally without internet access can achieve performance close to a larger vision-language model.

## 2 Related Work

Several previous works provide the foundation for the models used in this project. Rosenblatt introduced the perceptron as an early approach to pattern recognition and linear classification 1. Cox established statistical methods that form the foundation of logistic regression for classification 2. Cortes and Vapnik introduced Support Vector Machines (SVMs), which can classify data using nonlinear decision boundaries 3. LeCun et al. demonstrated the effectiveness of convolutional neural networks (CNNs) for image recognition, providing the foundation for our image-based CNN 4.

More directly related to this project, Tabak et al. applied machine learning to classify animal species in camera-trap images, demonstrating the potential of automated wildlife identification for ecological monitoring 5. Our project builds on this work by comparing multiple classical and neural models and evaluating whether lightweight models can approach the performance of a zero-shot vision-language model.

## 3 Methods

### 3.1 Dataset and Data Pre-processing

 *To be completed.*

### 3.2 Feature Extraction

 *To be completed.*

### 3.3 Classification Models

 *To be completed.*

### 3.4 Hyperparameters and Training

 *To be completed.*

### 3.5 Evaluation

 *To be completed.*

## 4 Preliminary Results

Initial experiments were conducted to establish baseline performance for the wildlife species classification task. The preliminary results below compare the models that have been implemented so far using the same test split and evaluation metrics. Accuracy and macro-F1 are reported to account for differences in class frequency across species.


| Model               | Accuracy | Macro-F1 |
| ------------------- | -------- | -------- |
| Perceptron          | --       | --       |
| Logistic Regression | --       | --       |
| SVM                 | --       | --       |
| CNN                 | --       | --       |
| Zero-Shot VLM       | --       | --       |


The preliminary results show **briefly describe which model performed best**. The **best-performing model** achieved an accuracy of **XX%** and a macro-F1 score of **XX**. The comparison between the classical models provides a baseline for evaluating whether the CNN and zero-shot vision-language model provide meaningful improvements in classification performance. These results are preliminary, and additional experimentation will focus on hyperparameter tuning, CNN training, and comparison with the zero-shot model.

## 5 Future Plan

*To be completed.*

## 6 References

### 1 The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain

**Author:** Frank Rosenblatt, 1958

 osenblatt introduced the perceptron as an early approach to pattern recognition and linear classification.

### 2 The Regression Analysis of Binary Sequences

**Author:** D. R. Cox, 1958

Cox established statistical methods that form the foundation of logistic regression for classification.

### 3 Support-Vector Networks

**Authors:** Corinna Cortes and Vladimir Vapnik, 1995

Cortes and Vapnik introduced Support Vector Machines as a classification method capable of finding effective decision    boundaries in high-dimensional feature spaces.

### 4 Gradient-Based Learning Applied to Document Recognition

**Authors:** Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998

This work demonstrated the effectiveness of convolutional neural networks for image recognition and classification.

### 5 Machine Learning to Classify Animal Species in Camera Trap Images: Applications in Ecology

**Authors:** Michael A. Tabak et al., 2019

This work applied machine learning to wildlife species classification using camera-trap imagery and demonstrated the potential of automated wildlife identification for ecological monitoring.