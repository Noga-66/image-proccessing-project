# CS303 – Image Processing Project (Streamlit App)

## 📌 Overview

This project is an **interactive Image Processing application** built using **Python + Streamlit**.
All operations are implemented **manually at pixel level** (without relying on OpenCV built‑in functions), as required for academic purposes.

The app allows users to upload images and apply a wide range of image processing techniques through a clean web interface.

---

## 🚀 Live Demo

🔗 **Deployed App:** *(Add your Streamlit Cloud link here)*

---

## 🧠 Features

### 🔹 Basic Image Operations

* Image upload (RGB images)
* Manual RGB → Grayscale conversion
* Display images side‑by‑side

### 🔹 Point Operations

* Brightness adjustment (Add / Multiply)
* Darkness adjustment (Subtract / Divide)
* Negative (Inverse image)

### 🔹 Arithmetic Operations

* Image addition
* Image subtraction

### 🔹 Histogram Operations

* Histogram computation (manual)
* Histogram visualization
* Contrast stretching
* Histogram equalization

### 🔹 Linear Filters (Manual)

* Mean filter
* Gaussian filter
* Laplacian filter

### 🔹 Non‑Linear Filters

* Median filter
* Min filter
* Max filter
* Range filter
* Mode filter

### 🔹 Noise Generation

* Salt & Pepper noise
* Gaussian noise
* Periodic noise

### 🔹 Morphological Operations

* Dilation
* Erosion
* Opening
* Closing

### 🔹 Segmentation & Dithering

* Automatic thresholding
* Floyd–Steinberg dithering

---

## 🛠️ Technologies Used

* **Python 3**
* **Streamlit** (UI & deployment)
* **NumPy** (manual pixel operations)
* **Matplotlib** (histogram visualization)
* **scikit‑image** (image I/O & resizing only)

---

## 📂 Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```
## team work
 * 1- Nada Hossam
 * 2-Alaa madeh
 * 3-Mina Mahmoud
 * 4-Mohamed Alaa
 * 5-Ahmed Reda
 * 6-Mazen Ahmed



✨ *This project demonstrates a strong understanding of image processing fundamentals and manual pixel manipulation.*
