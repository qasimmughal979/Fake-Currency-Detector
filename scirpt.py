import cv2
import numpy as np
import os
from tqdm import tqdm

def get_stats(folder):
    brightness = []
    contrast = []
    blur = []
    widths = []
    heights = []

    for file in tqdm(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # brightness: mean pixel value
        brightness.append(gray.mean())

        # contrast: std deviation
        contrast.append(gray.std())

        # blur: Laplacian variance
        blur.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        # resolution
        h, w = gray.shape
        widths.append(w)
        heights.append(h)

    return brightness, contrast, blur, widths, heights

real_stats = get_stats("Final_Clean_Dataset/Real")
fake_stats = get_stats("Final_Clean_Dataset/Fake")

print("\n--- REAL NOTES ---")
print("Brightness mean:", np.mean(real_stats[0]))
print("Contrast mean:", np.mean(real_stats[1]))
print("Blur mean:", np.mean(real_stats[2]))
print("Width mean:", np.mean(real_stats[3]))
print("Height mean:", np.mean(real_stats[4]))

print("\n--- FAKE NOTES ---")
print("Brightness mean:", np.mean(fake_stats[0]))
print("Contrast mean:", np.mean(fake_stats[1]))
print("Blur mean:", np.mean(fake_stats[2]))
print("Width mean:", np.mean(fake_stats[3]))
print("Height mean:", np.mean(fake_stats[4]))
