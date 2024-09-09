# SSD MobileNet Implementation for Yellowfin Tuna Detection
## Introduction
Illegal, unreported, and unregulated (IUU) fishing poses significant challenges to the sustainability of marine ecosystems, and one of the major issues is the overfishing of species like Yellowfin tuna. Traditional methods of monitoring fishing activities can be slow, labor-intensive, and prone to errors.

### Problem: To ensure sustainable fishing practices and reduce IUU, we need automated, scalable solutions that can detect and classify species like Yellowfin tuna in real-time.

### Solution: In this project, we use a Single Shot MultiBox Detector (SSD) combined with the MobileNet architecture to create a robust fish detection system. The SSD MobileNet model is highly optimized for speed and accuracy, making it an ideal choice for real-time applications with limited computational resources.

## Table of Contents
-Introduction
-Installation
-Dataset
-Training Process
-Evaluation
-Results
-Conclusion
## Installation
To run this project, you will need to install the following dependencies:

pip install tensorflow opencv-python numpy matplotlib
### Clone the repository:

git clone https://github.com/Ayoub-fox/Train_SSDmodelOnCustomDataset.git
cd ssd-mobilenet-yellowfin-tuna
Make sure to install other required libraries by following the steps on the notebook

## Dataset
For this project, the dataset consists of images of Yellowfin tuna. The data is divided into two sets:

Training Set: Used to train the model.
Testing Set: Used to evaluate model performance.
Annotations for each image are provided in the form of bounding boxes. The dataset follows the Pascal VOC format for object detection.

## Training Process
The SSD MobileNet architecture is a lightweight deep neural network designed for object detection. It works by predicting object classes and bounding boxes directly from the images in a single forward pass. The MobileNet backbone ensures that the model is fast and optimized for low-power devices like smartphones and embedded systems.

### To train the model:

Pre-process the images: Resize the images and annotations to the required input size.
Load the SSD MobileNet architecture from TensorFlow's Object Detection API.
Start training using the pre-processed data.
Model Architecture:
Base Network: MobileNet
Detection Method: SSD (Single Shot MultiBox Detector)
Training parameters include:

Batch Size: batch_size_value
Number of Epochs: num_epochs_value
Learning Rate: learning_rate_value
Training Command is on the .ypnb file copy it and paste it into your command prompt
![Capture](https://github.com/user-attachments/assets/7d9c9237-621d-4e90-85a9-b1c368814e28)

Evaluation
Once the model is trained, evaluate it on the test dataset. Performance metrics include:

mAP (Mean Average Precision): Measures the accuracy of the model in detecting objects.
Precision and Recall: To assess the model's ability to detect Yellowfin tuna accurately.
Evaluation Command:
bash
Copy code
python evaluate_model.py --model_path path_to_trained_model --test_data_path path_to_test_data
Results
The SSD MobileNet model was able to achieve strong results in detecting Yellowfin tuna with:

mAP Score: map_score_value
Precision: precision_value
Recall: recall_value
Placeholder: Add images showcasing detection results (e.g., bounding boxes drawn on the images of Yellowfin tuna).

Conclusion
The SSD MobileNet implementation provided an efficient and accurate solution for detecting Yellowfin tuna in images, showcasing its potential in addressing challenges related to sustainable fishing practices. The model's speed and low computational cost make it ideal for real-time monitoring systems in resource-constrained environments.

For future improvements, we could explore combining SSD MobileNet with other architectures like Faster R-CNN or YOLO to further enhance the accuracy of the detection system.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any inquiries or collaboration opportunities, please contact:

Your Name: YourEmail@example.com
