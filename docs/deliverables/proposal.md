# Project Proposal

## 1. Project Title

**TrailCam: Wildlife Species Recognition from Camera Trap Imagery**

## 2. Team Members

- Tyler Mestery

## 3. Project Details

### 3.1 Project Objective

The objective of this project is to build a wildlife species classifier for camera-trap imagery and evaluate whether a lightweight, non-LLM model can match the accuracy of a large vision-language model closely enough to be practical for field deployment, where internet connectivity and compute are usually unavailable.

Camera traps are motion-triggered cameras deployed across remote habitats to monitor wildlife populations, and conservation researchers currently must review thousands of images by hand to identify species, creating a backlog that slows down monitoring data becoming actionable. This backlog matters because timely, accurate species counts directly inform real conservation decisions, such as tracking population trends, detecting the presence of endangered or invasive species, and allocating limited habitat-protection resources, so delays in processing camera-trap data translate directly into delays in conservation response.

Machine learning is well suited to this problem because distinguishing visually similar species from motion-blurred, partially occluded, or poorly lit camera-trap photos requires learning subtle visual patterns rather than following fixed rules, and because a model running directly on a camera trap in the field cannot rely on a network connection to call a large hosted model.

### 3.2 Datasets

The dataset is **ENA24 (Eastern North America Camera Trap Detection Dataset)**, originally collected by the University of Missouri and distributed through LILA BC (Labeled Information Library of Alexandria: Biology and Conservation), the standard public repository for camera-trap research data.

- [LILA BC — ENA24 Detection Dataset](https://lila.science/datasets/ena24detection/)
- [Hugging Face — ENA24 Detection Dataset](https://huggingface.co/datasets/davanstrien/ena24-detection)

The dataset can also be loaded directly with the `datasets` library.

The images were collected by motion-triggered camera traps deployed at forested field sites across eastern North America, which automatically capture a photo whenever movement is detected within the camera’s field of view, day or night, over an extended deployment period.

The dataset contains **8,789 camera-trap images** at **1920×1080 resolution** with **9,772 bounding-box annotations** across **23 wildlife species** (e.g., American black bear, bobcat, eastern cottontail, coyote), labeled in COCO format under the **CDLA Permissive 1.0** license.

Each bounding box will be cropped into an individual labeled animal image. The label is the species class, and features are:

- **Classical models:** Color/texture descriptors
- **Convolutional and vision-language models:** Raw pixels

Images will be split roughly as follows:

- **70%** training
- **15%** validation
- **15%** testing

The split will occur at the original-image level to avoid placing near-duplicate crops from the same photo across different splits.

### 3.3 Machine Learning Algorithm

Four model families will be trained and evaluated on the cropped-animal species-classification task:

1. **Perceptron** — linear baseline using engineered color/texture features
2. **Logistic Regression** — linear baseline using engineered color/texture features
3. **Support Vector Machine (SVM)** — non-linear kernel using engineered color/texture features
4. **Convolutional Neural Network (CNN)** — trained directly on raw pixels

A **pretrained multimodal LLM** will also be prompted zero-shot to classify the same cropped images, with no task-specific training.

This progression is chosen not just to compare raw accuracy, but to answer a practical deployment question: whether a lightweight classical or CNN model, which can run directly on inexpensive field hardware with no internet connection, comes close enough to the accuracy of a zero-shot LLM (which would require connectivity and per-image API cost) to be the more sensible choice for real camera-trap deployments.

### 3.4 Expected Outcomes

I expect the **CNN** to achieve the highest classification accuracy and **macro-F1 score** (used because species counts are imbalanced across classes) among the models trained specifically on this dataset, with the **SVM** outperforming logistic regression and the perceptron.

I expect the **zero-shot LLM** to perform competitively on visually distinct species (e.g., American black bear) but to show a measurably lower F1 score on visually similar or rarer species that benefit from task-specific training.

The key expected outcome is a direct, quantified **accuracy-and-F1 comparison across all five models** on the held-out test split. Even if the LLM achieves the highest raw accuracy, a much cheaper classical or CNN model that scores within a few percentage points would be considered practically viable for actual field deployment, where connectivity and cost are real constraints.

The final deliverable is a **trained species classifier** along with this **comparative analysis**.
