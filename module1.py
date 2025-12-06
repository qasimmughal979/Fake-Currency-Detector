import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Configuration
DATASET_REAL = "Final_Clean_Dataset/Real"
DATASET_FAKE = "Final_Clean_Dataset/Fake"
OUTPUT_DIR = "Module1_Results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_sample_images(folder, num_samples=3):
    images = []
    filenames = []
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return [], []
        
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    # Select random samples or first few
    selected_files = files[:num_samples]
    
    for f in selected_files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            filenames.append(f)
    return images, filenames

# 1. Geometric Transformations
def apply_geometric(image, type="rotate", param=0):
    h, w = image.shape[:2]
    if type == "rotate":
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, param, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    elif type == "scale":
        return cv2.resize(image, None, fx=param, fy=param, interpolation=cv2.INTER_LINEAR)
    return image

# 2. Intensity Transformations
def apply_intensity(image, type="gamma", param=1.0):
    if type == "gamma":
        # Gamma correction: O = I^(1/gamma)
        invGamma = 1.0 / param
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    elif type == "equalize":
        # Histogram Equalization (YUV or Grayscale)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image

# 3. Noise Modeling
def add_noise(image, type="gaussian", param=0.1):
    if type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = param * 255 # param is percentage of max intensity
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype("uint8")
    elif type == "salt_pepper":
        s_vs_p = 0.5
        amount = param
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        return out
    return image

# 4. Restoration (Filtering)
def apply_restoration(image, type="gaussian", param=3):
    if type == "gaussian":
        k = int(param)
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(image, (k, k), 0)
    elif type == "median":
        k = int(param)
        if k % 2 == 0: k += 1
        return cv2.medianBlur(image, k)
    elif type == "bilateral":
        # d, sigmaColor, sigmaSpace
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

# 5. Edge Extraction
def extract_edges(image, type="canny", param1=100, param2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if type == "canny":
        return cv2.Canny(gray, param1, param2)
    elif type == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return gray

# 6. Feature Stability (ORB Keypoints)
def count_keypoints(image):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    return len(kp), kp

# 7. Frequency Domain Analysis (FFT)
def analyze_frequency(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def run_pipeline():
    print("Loading samples...")
    real_imgs, real_names = load_sample_images(DATASET_REAL, 1)
    fake_imgs, fake_names = load_sample_images(DATASET_FAKE, 1)
    
    if not real_imgs or not fake_imgs:
        print("Error: Could not load images from both datasets.")
        return

    real_img = real_imgs[0]
    fake_img = fake_imgs[0]
    
    print(f"Comparing Real ({real_names[0]}) vs Fake ({fake_names[0]})...")
    
    # --- Comparison 1: Edge Analysis ---
    real_edges = extract_edges(real_img, "canny")
    fake_edges = extract_edges(fake_img, "canny")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1); plt.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)); plt.title("Real Image")
    plt.subplot(2, 2, 2); plt.imshow(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)); plt.title("Fake Image")
    plt.subplot(2, 2, 3); plt.imshow(real_edges, cmap='gray'); plt.title("Real Edges")
    plt.subplot(2, 2, 4); plt.imshow(fake_edges, cmap='gray'); plt.title("Fake Edges")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Compare_Edges.png"))
    print("Saved Compare_Edges.png")

    # --- Comparison 2: Frequency Domain (FFT) ---
    real_fft = analyze_frequency(real_img)
    fake_fft = analyze_frequency(fake_img)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(real_fft, cmap='gray'); plt.title("Real Magnitude Spectrum")
    plt.subplot(1, 2, 2); plt.imshow(fake_fft, cmap='gray'); plt.title("Fake Magnitude Spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Compare_Frequency.png"))
    print("Saved Compare_Frequency.png")

    # --- Comparison 3: Feature Stability (Keypoints) ---
    k_r, kp_r = count_keypoints(real_img)
    k_f, kp_f = count_keypoints(fake_img)
    
    img_r_kp = cv2.drawKeypoints(real_img, kp_r, None, color=(0,255,0))
    img_f_kp = cv2.drawKeypoints(fake_img, kp_f, None, color=(0,0,255))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img_r_kp, cv2.COLOR_BGR2RGB)); plt.title(f"Real Keypoints: {k_r}")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(img_f_kp, cv2.COLOR_BGR2RGB)); plt.title(f"Fake Keypoints: {k_f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Compare_Features.png"))
    print("Saved Compare_Features.png")
    
    # --- Comparison 4: Response to Noise ---
    # Add noise to both and see which one loses more edges/details
    real_noisy = add_noise(real_img, "gaussian", 0.1)
    fake_noisy = add_noise(fake_img, "gaussian", 0.1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(real_noisy, cv2.COLOR_BGR2RGB)); plt.title("Real + Noise")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(fake_noisy, cv2.COLOR_BGR2RGB)); plt.title("Fake + Noise")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Compare_Noise_Response.png"))
    print("Saved Compare_Noise_Response.png")

    # --- Conclusion Graph ---
    generate_conclusion_graph()

def calculate_metrics(image):
    # 1. Edge Density
    edges = extract_edges(image, "canny")
    edge_density = np.sum(edges > 0) / edges.size
    
    # 2. Frequency Magnitude Mean
    fft_mag = analyze_frequency(image)
    freq_mean = np.mean(fft_mag)
    
    # 3. Keypoint Count
    kp_count, _ = count_keypoints(image)
    
    # 4. Blurriness (Laplacian Variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "Edge Density": edge_density,
        "Freq Mean": freq_mean,
        "Keypoints": kp_count,
        "Blurriness": blur_var
    }

def generate_conclusion_graph():
    print("Generating Conclusion Graph (processing 50 samples each)...")
    real_imgs, _ = load_sample_images(DATASET_REAL, 50)
    fake_imgs, _ = load_sample_images(DATASET_FAKE, 50)
    
    real_metrics = {"Edge Density": [], "Freq Mean": [], "Keypoints": [], "Blurriness": []}
    fake_metrics = {"Edge Density": [], "Freq Mean": [], "Keypoints": [], "Blurriness": []}
    
    for img in real_imgs:
        m = calculate_metrics(img)
        for k, v in m.items():
            real_metrics[k].append(v)
            
    for img in fake_imgs:
        m = calculate_metrics(img)
        for k, v in m.items():
            fake_metrics[k].append(v)
            
    # Calculate averages
    avg_real = {k: np.mean(v) for k, v in real_metrics.items()}
    avg_fake = {k: np.mean(v) for k, v in fake_metrics.items()}
    
    print("\n--- CONCLUSION METRICS (Average over 50 samples) ---")
    print(f"{'Metric':<20} | {'Real':<10} | {'Fake':<10}")
    print("-" * 46)
    for k in avg_real.keys():
        print(f"{k:<20} | {avg_real[k]:<10.4f} | {avg_fake[k]:<10.4f}")
    print("-" * 46)
    
    # Normalize for visualization (since scales are very different)
    # We will plot 4 separate subplots
    metrics = list(avg_real.keys())
    
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [avg_real[metric], avg_fake[metric]]
        plt.bar(["Real", "Fake"], values, color=['blue', 'red'])
        plt.title(f"Average {metric}")
        plt.ylabel("Value")
        
        # Add text labels
        for j, v in enumerate(values):
            plt.text(j, v, f"{v:.2f}", ha='center', va='bottom')
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Conclusion_Graph.png"))
    print("Saved Conclusion_Graph.png")

if __name__ == "__main__":
    run_pipeline()
