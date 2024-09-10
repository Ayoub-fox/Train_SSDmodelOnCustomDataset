# SSD MobileNet Implementation for Yellowfin Tuna Detection
## Introduction
Illegal, unreported, and unregulated (IUU) fishing poses significant challenges to the sustainability of marine ecosystems, and one of the major issues is the overfishing of species like Yellowfin tuna. Traditional methods of monitoring fishing activities can be slow, labor-intensive, and prone to errors.

### Problem:
To ensure sustainable fishing practices and reduce IUU, we need automated, scalable solutions that can detect and classify species like Yellowfin tuna in real-time.

### Solution:
In this project, we use a Single Shot MultiBox Detector (SSD) combined with the MobileNet architecture to create a robust fish detection system. The SSD MobileNet model is highly optimized for speed and accuracy, making it an ideal choice for real-time applications with limited computational resources.

## Table of Contents
-Introduction<br>
-Installation<br>
-Dataset<br>
-Training Process<br>
-Evaluation<br>
-Results<br>
-Conclusion<br>
## Installation
To run this project, you will need to install the following dependencies:

pip install tensorflow opencv-python numpy matplotlib
### Clone the repository:

git clone https://github.com/Ayoub-fox/Train_SSDmodelOnCustomDataset.git<br>
cd ssd-mobilenet-yellowfin-tuna<br>
Make sure to install other required libraries by following the steps on the notebook

## Dataset
For this project, the dataset consists of images of Yellowfin tuna. The data is divided into two sets:

Training Set: Used to train the model.<br>
Testing Set: Used to evaluate model performance.<br>
Annotations for each image are provided in the form of bounding boxes. The dataset follows the Pascal VOC format for object detection.

## Training Process
The SSD MobileNet architecture is a lightweight deep neural network designed for object detection. It works by predicting object classes and bounding boxes directly from the images in a single forward pass. The MobileNet backbone ensures that the model is fast and optimized for low-power devices like smartphones and embedded systems.

### To train the model:

Pre-process the images: Resize the images and annotations to the required input size.<br>
Load the SSD MobileNet architecture from TensorFlow's Object Detection API.<br>
Start training using the pre-processed data.<br>
Model Architecture:<br>
Base Network: MobileNet<br>
Detection Method: SSD (Single Shot MultiBox Detector)<br>
Training parameters include:<br>

Batch Size: batch_size_value<br>
Number of Epochs: num_epochs_value<br>
Learning Rate: learning_rate_value<br>
Training Command is on the .ypnb file copy it and paste it into your command prompt

![Capture1](https://github.com/user-attachments/assets/b1a77ff0-b86b-4cd4-81fa-3db5da814367) ![Capture2](https://github.com/user-attachments/assets/df0e943b-0a65-4bb8-bd10-a3f30e7c9253)

## Evaluation
Once the model is trained, evaluate it on the test dataset. Performance metrics include:

mAP (Mean Average Precision): Measures the accuracy of the model in detecting objects.<br>
Precision and Recall: To assess the model's ability to detect Yellowfin tuna accurately.<br>
Evaluation Command: 
python path_to_your_TensorFlow_models/research/object_detection/model_main_tf2.py \
--model_dir=path_to_your_model_directory/my_ssd_mobnet_640 \
--pipeline_config_path=path_to_your_model_directory/my_ssd_mobnet_640/pipeline.config \
--checkpoint_dir= path\to\your\model\chekpoints
### Output :

![Capture](https://github.com/user-attachments/assets/7d9c9237-621d-4e90-85a9-b1c368814e28)

## Results
The SSD MobileNet model was able to achieve strong results in detecting Yellowfin tuna with:

mAP Score: 0.37

![imageData (3)](https://github.com/user-attachments/assets/99ec4a5c-bc34-479f-ad8a-37348c36b302)

## Conclusion
The SSD MobileNet implementation provided an efficient and accurate solution for detecting Yellowfin tuna in images, showcasing its potential in addressing challenges related to sustainable fishing practices. The model's speed and low computational cost make it ideal for real-time monitoring systems in resource-constrained environments.


Contact
For any inquiries or collaboration opportunities, please contact:

Your Name: ayoubbenachour77@gmail.com
