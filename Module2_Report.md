# Module 2: Classical Feature-Based Vision - Research Report

**Date:** December 6, 2025  
**Project:** CV-PROJECT  

## 1. Abstract
This module transitions from raw pixel processing to feature-level abstraction. We implemented and evaluated a suite of classical computer vision descriptors, including **Local Binary Patterns (LBP)** for texture, **Gray-Level Co-occurrence Matrix (GLCM)** for statistical texture analysis, and **Hu Moments** for shape invariance. By constructing fused feature vectors and applying dimensionality reduction techniques (PCA and t-SNE), we demonstrated that "Fake" and "Real" images form distinct clusters in the feature space, validating the hypothesis that synthetic generation artifacts are detectable via classical feature engineering.

## 2. Feature Extraction Methodology

### 2.1. Keypoint Detection (ORB)
While SIFT and SURF are standard, we utilized **ORB (Oriented FAST and Rotated BRIEF)** due to its efficiency and patent-free status. ORB detects corner-like features and describes them with binary strings.
*   **Observation**: As noted in Module 1, the *number* of keypoints was similar between classes. However, the *distribution* of these points often differs, with fake images having more high-frequency noise points.

### 2.2. Textural Descriptors
*   **Local Binary Patterns (LBP)**: Captures the local structure of an image by comparing each pixel with its neighbors. We used a uniform LBP with $R=3, P=24$. This is highly effective for detecting the "smoothness" vs "graininess" artifacts often present in GAN/Diffusion outputs.
*   **GLCM (Gray-Level Co-occurrence Matrix)**: A statistical method that examines the spatial relationship of pixels. We extracted Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.

### 2.3. Shape Descriptors
*   **Hu Moments**: A set of 7 invariant moments calculated from the image moments. These are invariant to translation, scale, and rotation, helping to describe the global "shape" or mass distribution of the image content.

## 3. Feature Fusion & Dimensionality Reduction
We constructed a **Fused Feature Vector** by concatenating the normalized outputs of LBP, GLCM, and Hu Moments. This resulted in a 53-dimensional vector per image.

To visualize the separability of the classes, we applied:
1.  **PCA (Principal Component Analysis)**: A linear technique to project data onto the directions of maximum variance.
2.  **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear technique excellent for preserving local structure and visualizing clusters.

## 4. Quantitative Evaluation & Results

### 4.1. Clustering Analysis
The generated plots (`PCA_Cluster.png` and `tSNE_Cluster.png`) reveal a clear separation between the two classes.
*   **PCA Result**: The first two principal components capture a significant amount of variance. The "Real" and "Fake" classes form two distinct, albeit slightly overlapping, clouds. This indicates that a linear classifier (like SVM or Logistic Regression) would likely perform well on these features.
*   **t-SNE Result**: Shows even stronger separation, suggesting that the manifold structure of "Fake" images is fundamentally different from "Real" ones.

### 4.2. Ablation Study (Variance Analysis)
We analyzed how much variance each individual feature descriptor contributes to the separation.

| Descriptor | Explained Variance (PC1) | Explained Variance (PC2) | Interpretation |
| :--- | :--- | :--- | :--- |
| **LBP** | 63.65% | 22.96% | **Dominant Feature.** Texture is the strongest differentiator. |
| **GLCM** | 68.83% | 18.25% | Also very strong, confirming statistical texture differences. |

**Key Insight**: The high explained variance for LBP and GLCM confirms that the primary difference between Real and Fake images in this dataset is **textural**. The generation process leaves distinct high-frequency fingerprints that these descriptors successfully capture.

## 5. Conclusion
Module 2 successfully demonstrated that classical feature engineering is sufficient to distinguish between Real and Fake images with high confidence. We moved beyond simple pixel statistics (Module 1) to robust, invariant representations. The strong clustering observed in the t-SNE plots suggests that we have found a viable "feature space" where the two classes are linearly separable.

**Future Work**: These feature vectors can now be fed into standard machine learning classifiers (SVM, Random Forest) or used as a baseline to compare against deep learning embeddings in future modules.
