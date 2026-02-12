# Coronary Artery Segmentation

A comprehensive implementation of coronary artery segmentation using multiple approaches: manual segmentation with 3D Slicer, traditional computer vision techniques with OpenCV, and deep learning with U-Net architecture.

## 📋 Project Overview

This project investigates different methodologies for segmenting coronary arteries from medical imaging data. Coronary artery disease is one of the leading causes of death worldwide, and accurate segmentation of coronary vessels is crucial for diagnosis and treatment planning.

### Key Objectives

1. **Manual Segmentation**: Perform manual segmentation of coronary vessels using 3D Slicer medical imaging software
2. **Traditional Computer Vision**: Implement and evaluate image processing techniques using OpenCV for automatic segmentation
3. **Deep Learning**: Develop and train U-Net models for automatic segmentation

## 📊 Dataset

This project uses the **ARCADE** (Automatic Region-based Coronary Artery Disease diagnostics) dataset, a publicly available collection containing:

- X-ray coronary angiography images
- Expert annotations of vessel structures
- Stenotic region labels

## 🛠️ Methods Implemented

### 1. OpenCV-Based Segmentation

**Workflow 1: Traditional Image Processing**

- Gaussian blur for noise reduction
- Contrast enhancement (Global and CLAHE)
- Adaptive thresholding
- Morphological transformations
- Connected Component Analysis (CCA)

**Workflow 2: Edge Detection**

- Canny edge detection
- Additional refinement techniques

### 2. Deep Learning with U-Net

- Custom U-Net architecture implementation
- Multiple downsampling level configurations (5 and 6 levels)
- Training with loss curve monitoring
- Validation on test dataset

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
OpenCV
TensorFlow/Keras or PyTorch
NumPy
Matplotlib
scikit-image
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coronary-artery-segmentation.git
cd coronary-artery-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### OpenCV Segmentation

```python
from src.opencv_methods import segment_coronary_arteries

# Load image
image = cv2.imread('path/to/angiography/image.png', 0)

# Perform segmentation
segmented = segment_coronary_arteries(image, workflow='clahe')

# Visualize results
plt.imshow(segmented, cmap='gray')
plt.show()
```

#### U-Net Segmentation

```python
from src.unet import UNet
from src.preprocessing import preprocess_image

# Load model
model = UNet.load('models/unet_best.h5')

# Preprocess and predict
image = preprocess_image('path/to/image.png')
prediction = model.predict(image)
```

## 🔬 Research Context

This work was completed as part of the **Biomedical Image Processing** course at the Faculty of Engineering, University of Kragujevac, for the EMMBIOME Master's program (School year 2025/2026).

**Student**: Minh Triet Ho  
**Course**: Biomedical Image Processing  
**Date**: 02/2026
