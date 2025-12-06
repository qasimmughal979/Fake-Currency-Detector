import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Configuration
DATASET_REAL = "Final_Clean_Dataset/Real"
DATASET_FAKE = "Final_Clean_Dataset/Fake"
OUTPUT_DIR = "Module2_Results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_images(folder, limit=100):
    images = []
    labels = []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    # Take a subset for efficiency
    selected_files = files[:limit]
    
    for f in selected_files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is not None:
            # Resize for consistency and speed
            img = cv2.resize(img, (256, 256))
            images.append(img)
            labels.append(f)
    return images, labels

# --- 1. Keypoint Detection (ORB as SIFT/SURF are patented/slower) ---
def extract_orb_features(image):
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(image, None)
    if des is None:
        return np.zeros(32 * 500) # Return zero vector if no features
    # Flatten or use Bag of Words later. For now, just return descriptor stats
    return des.flatten()[:500] # Truncate/Pad for simple vector

# --- 2. Textural Descriptors ---

# LBP (Local Binary Patterns) - Texture
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# HOG (Histogram of Oriented Gradients) - Shape
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

# GLCM (Gray-Level Co-occurrence Matrix) - Statistical Texture
def extract_glcm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Quantize to speed up GLCM
    gray = (gray // 4).astype(np.uint8) 
    glcm = graycomatrix(gray, distances=[1, 5], angles=[0, np.pi/2], levels=64, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

# Hu Moments - Shape Invariants
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform to make them comparable
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
    return hu_moments

# --- 3. Feature Fusion & Vector Construction ---
def extract_feature_vector(image):
    # Combine multiple descriptors
    lbp = extract_lbp(image)
    glcm = extract_glcm(image)
    hu = extract_hu_moments(image)
    
    # HOG is very high dimensional, maybe too much for simple fusion without reduction first
    # Let's stick to Texture + Shape stats for the "Fused" vector
    return np.hstack([lbp, glcm, hu])

def run_module2():
    print("Loading dataset (100 samples each)...")
    real_imgs, _ = load_images(DATASET_REAL, 100)
    fake_imgs, _ = load_images(DATASET_FAKE, 100)
    
    if not real_imgs or not fake_imgs:
        print("Error loading images.")
        return

    print("Extracting features...")
    
    # Feature Matrix: Rows = Samples, Cols = Features
    X = []
    y = [] # 0 for Real, 1 for Fake
    
    for img in tqdm(real_imgs, desc="Real"):
        vec = extract_feature_vector(img)
        X.append(vec)
        y.append(0)
        
    for img in tqdm(fake_imgs, desc="Fake"):
        vec = extract_feature_vector(img)
        X.append(vec)
        y.append(1)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature Vector Shape: {X.shape}")
    
    # --- 4. Dimensionality Reduction & Visualization ---
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Real', alpha=0.6)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Fake', alpha=0.6)
    plt.title("PCA of Fused Features (LBP + GLCM + Hu)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "PCA_Cluster.png"))
    print("Saved PCA_Cluster.png")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='blue', label='Real', alpha=0.6)
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='red', label='Fake', alpha=0.6)
    plt.title("t-SNE of Fused Features (LBP + GLCM + Hu)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "tSNE_Cluster.png"))
    print("Saved tSNE_Cluster.png")
    
    # --- 5. Ablation / Individual Feature Analysis ---
    # Let's see which feature type separates best on its own (using PCA variance as proxy)
    
    print("Analyzing individual feature contributions...")
    
    # Extract just LBP for all
    X_lbp = [extract_lbp(img) for img in real_imgs + fake_imgs]
    X_lbp = scaler.fit_transform(X_lbp)
    pca_lbp = PCA(n_components=2)
    pca_lbp.fit(X_lbp)
    print(f"LBP Explained Variance: {pca_lbp.explained_variance_ratio_}")
    
    # Extract just GLCM
    X_glcm = [extract_glcm(img) for img in real_imgs + fake_imgs]
    X_glcm = scaler.fit_transform(X_glcm)
    pca_glcm = PCA(n_components=2)
    pca_glcm.fit(X_glcm)
    print(f"GLCM Explained Variance: {pca_glcm.explained_variance_ratio_}")

if __name__ == "__main__":
    run_module2()
