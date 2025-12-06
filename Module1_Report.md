# Module 1: Foundations of Vision and Image Analysis - Research Report

**Date:** December 5, 2025  
**Project:** CV-PROJECT  

## 1. Abstract
This report documents the implementation and findings of Module 1, which focuses on establishing a foundational visual processing pipeline. We analyzed a dataset of Real and Fake images, applied various geometric and intensity transformations, and evaluated the images using tunable mathematical functions. A comparative analysis was conducted to quantify differences in edge density, frequency domain characteristics, and feature stability between real and synthetic imagery.

## 2. Dataset Characterization
The dataset consists of two distinct classes: **Real** and **Fake**. Initial statistical analysis reveals significant differences in the fundamental properties of the images.

### 2.1. Statistical Overview
Based on an analysis of the full dataset, the following average characteristics were observed:

| Feature | Real Images (Mean) | Fake Images (Mean) | Observation |
| :--- | :--- | :--- | :--- |
| **Brightness** | ~88.30 | ~114.04 | Fake images are generally brighter. |
| **Contrast** | ~56.59 | ~67.25 | Fake images exhibit higher contrast. |
| **Blur (Laplacian Var)** | ~163.02 | ~460.64 | Fake images appear significantly sharper (or have more high-frequency noise). |
| **Resolution** | ~1021 x 1061 | ~3402 x 3348 | Fake images are much higher resolution. |

**Key Insight:** The "Fake" dataset contains images that are not only larger but also statistically distinct in terms of pixel intensity distribution and sharpness. This suggests that the generation process (likely GAN or Diffusion based) tends to produce high-contrast, high-frequency artifacts.

## 3. Preprocessing Methodology
To evaluate image interpretability and robustness, we implemented a pipeline consisting of the following mathematical transformations:

### 3.1. Geometric Transformations
*   **Rotation**: Implemented using affine transformation matrices (`cv2.getRotationMatrix2D`).
*   **Scaling**: Implemented using bicubic interpolation.
*   **Purpose**: To test the stability of feature detectors (ORB) under geometric perturbations.

### 3.2. Intensity Transformations
*   **Gamma Correction**: Non-linear operation ($O = I^{1/\gamma}$) to adjust luminance.
*   **Histogram Equalization**: Applied to the Y-channel of the YUV color space to enhance contrast without shifting color balance.

### 3.3. Noise Modeling & Restoration
*   **Noise Injection**: Gaussian noise ($\mu=0, \sigma=var$) and Salt-and-Pepper noise were modeled to simulate sensor degradation.
*   **Restoration**:
    *   **Gaussian Blur**: Linear smoothing.
    *   **Median Filter**: Non-linear filtering effective against salt-and-pepper noise.
    *   **Bilateral Filter**: Edge-preserving smoothing.

### 3.4. Edge Extraction
*   **Canny Edge Detector**: Multi-stage algorithm (Gaussian filter, gradient calculation, non-maximum suppression, hysteresis thresholding).
*   **Sobel Operator**: Gradient-based edge detection.

## 4. Comparative Analysis & Results
We conducted a comparative study on a subset of 50 images from each class to derive quantitative metrics.

### 4.1. Quantitative Metrics (Average over 50 samples)

| Metric | Real | Fake | Delta |
| :--- | :--- | :--- | :--- |
| **Edge Density** | 0.0502 | 0.0807 | **+60.7%** |
| **Frequency Mean** | 155.97 | 177.78 | **+14.0%** |
| **Keypoints (ORB)** | 496.30 | 499.86 | **+0.7%** |
| **Blurriness (Sharpness)**| 350.67 | 363.31 | **+3.6%** |

### 4.2. Analysis of Findings

#### Edge Density
The **Fake** images exhibit a significantly higher edge density (0.0807 vs 0.0502). This correlates with the higher contrast and sharpness observed in the dataset characterization. The generation process appears to introduce more high-frequency transitions, resulting in "busier" edge maps.

#### Frequency Domain (FFT)
The Frequency Mean is higher for Fake images (177.78 vs 155.97). In the frequency domain visualizations (`Compare_Frequency.png`), synthetic images often display distinct spectral artifacts or a different energy distribution compared to natural images, which tend to follow a $1/f$ power law more closely.

#### Feature Stability
Interestingly, despite the resolution and sharpness differences, the number of detected **ORB Keypoints** is nearly identical (approx. 500 for both). This suggests that while Fake images have more *edges*, they do not necessarily contain more *distinctive features* (corners/blobs) that ORB detects, or that the detector saturates at the configured limit (default 500).

#### Robustness to Noise
Visual inspection of `Compare_Noise_Response.png` indicates that the higher contrast of Fake images makes them slightly more robust to Gaussian noise visually, as the signal-to-noise ratio remains higher due to the stronger underlying signal (pixel intensity).

## 5. Conclusion
Module 1 successfully established a visual processing pipeline and characterized the dataset. The most significant finding is the **spectral and structural disparity** between Real and Fake images. Fake images are characterized by:
1.  Higher contrast and brightness.
2.  Significantly higher edge density.
3.  Higher energy in the frequency domain.

These "fingerprints" suggest that simple filtering or frequency-based analysis could be effective early-stage classifiers for this specific dataset, even before applying deep learning techniques. The implemented pipeline allows for tunable experimentation with these parameters to further isolate these distinguishing features.
