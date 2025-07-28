# Eye_Color_Detection
**INTRODUCTION** 

Eye color detection is an important task in the field of image processing and computer 
vision. This project involves detecting the color of a person’s eye using Convolutional 
Neural Networks (CNNs). Eye color, a distinguishing biometric trait, can be categorized 
into several types such as brown, blue, green, gray etc. By training a CNN on a dataset of 
labeled eye color images, we can classify the input eye images into their respective color 
categories with high accuracy.

**METHODOLOGY**

3.1 Overview 
The project follows a standard machine learning pipeline that includes data collection, data 
preparation, preprocessing, and model training. Both a custom Convolutional Neural 
Network (CNN) model and Transfer Learning techniques using pre-trained architectures are 
employed to improve classification performance and generalization. 
3.1.1 Data Collection 
A labeled dataset containing images of eyes in different colors such as brown, blue, green, 
and gray is used. Each image is annotated with the correct eye color category to enable 
supervised learning. The dataset is collected from publicly available sources and manually 
curated collections to ensure diversity. It covers a range of variations in lighting, pose, and 
image quality to better reflect real-world conditions. 
3.1.2 Data Preparation 
The dataset is divided into training, validation, and testing subsets to properly evaluate 
model performance. All images are resized to a uniform dimension to maintain consistency 
and match model input requirements. Normalization is applied to scale pixel values between 
0 and 1. Data augmentation techniques such as rotation, flipping, and brightness changes are 
applied to the training set to enhance model generalization. 
3.1.3 Pre-Processing - Resizing images - Normalizing pixel values - Data augmentation (rotation, flipping) 
3.1.4 Convolutional Neural Networks (CNNs) 
CNNs are used to automatically learn and extract key features from eye images. Their 
layered structure is effective for image data, enabling accurate classification based on 
learned features. 
3.1.5 Transfer Learning 
To further enhance model performance, Transfer Learning is employed using pre-trained 
models such as VGG16 and ResNet50. These models, trained on large-scale datasets like 
ImageNet, are fine-tuned on the eye color dataset. This approach leverages learned features 
from generic image data and adapts them to the specific task of eye color classification, 
improving accuracy and reducing training time. 
3.2 Architecture of Proposed System 
The CNN and Transfer Learning models are trained on the dataset and tested on unseen 
data. Results show that the models achieve high accuracy in detecting eye color, with 
Transfer Learning models often outperforming baseline CNNs due to their ability to 
leverage pre-learned visual features.

