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

This project uses the ENA24 (Eastern North America Camera Trap Detection Dataset) from LILA BC, which contains 8,789 camera-trap images (1920×1080) with 9,772 COCO-format bounding-box annotations across 23 wildlife species. The dataset is loaded via the Hugging Face `datasets` library. For species classification, each bounding box is cropped from its source image to form a labeled animal image, where the label is the annotated species. The data are split into training (70%), validation (15%), and test (15%) sets at the original-image level so that all crops from the same photo remain in the same split, reducing leakage from near-duplicate views of the same scene. A fixed random seed (42) is used so the partitions are reproducible; all models are trained and evaluated on these same crop-level splits.

### 3.2 Feature Extraction

The perceptron, logistic regression, and SVM operate on fixed-length feature vectors rather than raw pixels. For each animal crop, an HSV color histogram captures species coloration with some robustness to day/night brightness changes, and Local Binary Pattern (LBP) texture features capture coat and body patterning under blur and occlusion. These descriptors are concatenated into one feature vector per crop, extracted once from the Section 3.1 images, and reused across classical models. The CNN and zero-shot vision-language model skip this pipeline and use crop pixels directly (resized and normalized for the CNN).

### 3.3 Classification Models

Five approaches are compared on the same labels and splits. The perceptron is a linear baseline that learns a separating hyperplane in feature space. Logistic regression is a linear, probabilistic classifier with L2 regularization for a stronger classical baseline. An RBF-kernel SVM adds nonlinear decision boundaries when species are not linearly separable in the engineered features. A CNN is trained end-to-end on crop pixels to learn spatial filters suited to animal appearance. A pretrained CLIP-style vision-language model is used zero-shot by matching each crop to the 23 species names as text prompts, with no ENA24 fine-tuning. This mix spans lightweight offline models and a larger pretrained baseline for deployment-oriented comparison.

### 3.4 Hyperparameters and Training

Classical models use scikit-learn on the training split. Logistic regression uses L2 regularization with the penalty strength `C` selected on the validation set. The SVM uses an RBF kernel with `C` and `gamma` chosen by validation search. The CNN resizes crops to 224×224, trains with Adam (learning rate 1e-3) and mini-batches of size 32, and stops early when validation macro-F1 stops improving. CLIP is not trained on ENA24; each test crop is scored once against a fixed set of species text prompts.

### 3.5 Evaluation

All models are evaluated on the held-out test split only. Predictions are compared to ground-truth species labels using accuracy and macro-averaged F1 score; macro-F1 treats each species equally and is reported because class frequencies in camera-trap data are imbalanced. The same test crops and metrics are used for every model so results reflect comparable classification performance rather than differences in data partitioning.

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

Through the rest of the project, I will finish the ENA24 data pipeline, extract HSV and LBP features, and train all five models (perceptron, logistic regression, SVM, CNN, and zero-shot CLIP) on the same splits. I will tune classical and CNN hyperparameters on the validation set, then report final test accuracy and macro-F1 for every model. The remaining work focuses on analyzing where lightweight offline models succeed or fail relative to CLIP, especially on rare or visually similar species, and writing up a clear recommendation for field deployment. The final deliverable is a complete comparative evaluation and a trained species classifier suitable for offline use.

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