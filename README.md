# Eye Color Detection

A Convolutional Neural Network (CNN) based system to classify human eye color into Blue, Brown, Gray, and Green from an image.

**Overview**

Eye color detection plays an important role in biometrics, security, healthcare, and human-computer interaction.

This project uses a CNN model (with optional Transfer Learning) to accurately classify eye colors from cropped eye images.


**Problem Statement**

Manual eye–color classification is subjective and slow.

The goal is to build an automated eye-color classifier that works reliably across variations like:

  Lighting
  
  Occlusion (hair, glasses)
  
  Low-resolution images
  
  Subtle color differences
  

**Motivation**

Eye color is a unique biometric trait. With the increasing use of facial analysis and recognition systems, automating eye-color detection can support:

  Identity verification
  
  Forensics
  
  Healthcare diagnostics
  
  Smart marketing
  
  Interactive applications
  

**Literature Review**

Modern research shows that CNNs outperform classical image-processing methods such as histogram and texture analysis.

Notable studies:

  Human Eye Color Classification Using CNN – Achieved high accuracy using custom CNN and strong data augmentation.
  
  Deep Learning for Eye Color Classification in the Wild – Used advanced preprocessing and augmentation to handle real-world images.
  
  These works validate CNN + Transfer Learning for eye-color prediction.


**Methodology**
  1. Data Collection
     
    Dataset includes labeled eye images across four classes:
    
      Blue, Brown, Gray, Green
      
    Images vary in lighting, resolution, and angle.
    
  2. Preprocessing
     
    Resize to 128 × 128
    
    Normalize pixel values
    
    Train/validation/test split
    
    Data augmentation:
    
      Flip
      
      Rotation
      
      Brightness/contrast
      
      Gamma adjustment
      
      Coarse dropout
      
  3. Model Architecture

    Two approaches:
     
      Custom CNN
     
        Input → Conv → Pool → Conv → Pool → Flatten → Dense → Softmax (4 classes)
     
      Transfer Learning
     
        Models used:
     
          VGG16
     
          ResNet50

**System Requirements**

  Hardware
  
    i5 processor (or higher)
    
    8–16 GB RAM
    
    Optional GPU for faster training
    
  Software

    Python 3.7+
    
    TensorFlow / Keras
    
    NumPy, Matplotlib
    
    OpenCV
    
    scikit-learn

**Training & Evaluation**

  Plots generated:
  
    Training accuracy vs validation accuracy
    
    Training loss vs validation loss
    
    Model evaluation includes:
    
      Overall accuracy
    
      Confusion matrix
      
      Testing on unseen images

**Eye Color Prediction (Example Usage)**

    test_image = cv2.imread("sample.jpg")
  
    predicted_color = predict_eye_color_from_array(test_image)
  
    print("Predicted Eye Color:", predicted_color)
  
    Output:
      Predicted Eye Color: Brown

**Results**

The CNN achieved strong classification accuracy, and Transfer Learning models further improved performance in noisy or low-light images.

**Future Scope**

  Expand dataset with more diversity
  
  Add real-time detection with TensorFlow Lite / OpenCV
  
  Use advanced models (EfficientNet, ViT)
  
  Improve augmentation and preprocessing pipelines

**References**
  Goodfellow et al., Deep Learning
  Krizhevsky et al., ImageNet Classification
  Bhargavi & Pranathi, CNN for Eye Color Classification
  Zhao et al., Eye Color Classification in the Wild
