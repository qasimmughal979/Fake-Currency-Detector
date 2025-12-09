# CurrencyGuard: AI-Based Fake Currency Detection System

An advanced computer vision system designed to authenticate currency notes using deep learning and generative AI. This project combines a **ResNet50** classification model with **Google's Gemini Vision API** to verify verification and detect fake Pakistani currency notes.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)

## 🚀 Features

-   **Dual-Stage Verification:**
    1.  **Presence Check:** Uses **Gemini Vision API** to verify if the image contains a valid Pakistani currency note.
    2.  **Authenticity Check:** Uses a fine-tuned **ResNet50** model to classify the note as "Real" or "Fake".
-   **Modern Web Interface:** Built with **Next.js**, **Tailwind CSS**, and **Framer Motion** for a smooth user experience.
-   **Secure Processing:** Backend running on **Flask** with secure API key management using `.env`.
-   **Real-Time Analysis:** Provides instant feedback with confidence scores.

## 🛠️ Tech Stack

### Backend
-   **Framework:** Flask (Python)
-   **ML Model:** TensorFlow/Keras (ResNet50)
-   **Verification:** Google Gemini Vision API (`gemini-2.5-flash`)
-   **Image Processing:** OpenCV, PIL, NumPy

### Frontend
-   **Framework:** Next.js 16 (React)
-   **Styling:** Tailwind CSS, ShadCN UI
-   **Animations:** Framer Motion
-   **Icons:** Lucide React

## 📂 Project Structure

```
Fake-Currency-Detector/
├── backend/                # Flask Backend
│   ├── app.py              # Main API entry point
│   ├── gemini_verifier.py  # Gemini AI verification module
│   ├── requirements.txt    # Python dependencies
│   └── test_gemini.py      # detailed testing script
├── web-app/                # Next.js Frontend
│   ├── src/app/page.tsx    # Main UI page
│   ├── src/components/     # UI Components (Header, Footer)
│   └── package.json        # Frontend dependencies
├── module3.py              # Transfer Learning Training Script
└── augment_fake_data.py    # Data Augmentation Script
```

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Fake-Currency-Detector.git
cd Fake-Currency-Detector
```

### 2. Backend Setup
1.  Navigate to the project root.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables:
    -   Create a `.env` file in the `backend/` folder.
    -   Add your Gemini API Key:
        ```bash
        GEMINI_API_KEY="your_api_key_here"
        ```
4.  Run the backend:
    ```bash
    python3 backend/app.py
    ```
    *Server runs on http://127.0.0.1:5000*

### 3. Frontend Setup
1.  Navigate to the frontend folder:
    ```bash
    cd web-app
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
    *App runs on http://localhost:3000*

## 📖 Usage

1.  Open the web app at `localhost:3000`.
2.  Upload an image of a Pakistani currency note.
3.  The system will:
    -   First verify if it is a Pakistani note (using Gemini).
    -   If verified, analyze it for authenticity (Real/Fake).
4.  View the result and confidence score.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

This project is licensed under the MIT License.
