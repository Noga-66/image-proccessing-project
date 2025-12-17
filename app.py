
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import random
import math

# Initialize session state for images
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'img2' not in st.session_state:
    st.session_state.img2 = None

# Utility functions
def ensure_gray(img):
    """Convert image to grayscale manually"""
    if len(img.shape) == 3:
        return ((img[:, :, 0].astype(int) + img[:, :, 1].astype(int) + img[:, :, 2].astype(int)) // 3).astype(np.uint8)
    return img

def display_images(img1, img2=None, title1="Original", title2="Processed"):
    """Display images side by side"""
    if img2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=title1, use_column_width=True)
        with col2:
            st.image(img2, caption=title2, use_column_width=True)
    else:
        st.image(img1, caption=title1, use_column_width=True)

def compute_hist(img):
    hist = np.zeros(256, dtype=int)
    for p in img.flatten():
        hist[p] += 1
    return hist

def show_histogram(img):
    hist = compute_hist(img)
    fig, ax = plt.subplots()
    ax.bar(range(256), hist)
    ax.set_title("Histogram")
    st.pyplot(fig)

# Streamlit App
st.title("CS303 Image Processing - Manual Pixel Operations")

# Basic IO & Color
st.header("Basic IO & Color")
uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="upload1")
if uploaded_file is not None:
    st.session_state.original_image = io.imread(uploaded_file)
    display_images(st.session_state.original_image)

uploaded_file2 = st.file_uploader("Upload 2nd Image", type=['png', 'jpg', 'jpeg'], key="upload2")
if uploaded_file2 is not None and st.session_state.original_image is not None:
    img2 = io.imread(uploaded_file2)
    img2 = transform.resize(img2, st.session_state.original_image.shape[:2], preserve_range=True).astype(np.uint8)
    st.session_state.img2 = img2
    display_images(st.session_state.img2, title1="Second Image")

if st.button("RGB to Gray (Manual)"):
    if st.session_state.original_image is not None:
        st.session_state.processed_image = ensure_gray(st.session_state.original_image)
        display_images(st.session_state.original_image, st.session_state.processed_image, "RGB", "Gray")

# Point Operations
st.header("Point Operations")
brightness_op = st.selectbox("Brightness Operation", ["add", "mult"], key="bright_op")
brightness_val = st.number_input("Value", value=0.0, key="bright_val")
if st.button("Apply Brightness"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        if brightness_op == "add":
            res = img + brightness_val
        else:
            res = img * brightness_val
        st.session_state.processed_image = np.clip(res, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", f"Brightness ({brightness_op})")

darkness_op = st.selectbox("Darkness Operation", ["sub", "div"], key="dark_op")
darkness_val = st.number_input("Value", value=0.0, key="dark_val")
if st.button("Apply Darkness"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        if darkness_op == "sub":
            res = img - darkness_val
        else:
            res = np.where(darkness_val != 0, img / darkness_val, img)
        st.session_state.processed_image = np.clip(res, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", f"Darkness ({darkness_op})")

if st.button("Inverse (Negative)"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        st.session_state.processed_image = 255 - img
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", "Negative")

# Arithmetic Operations
st.header("Arithmetic Operations")
if st.button("Add Two Images"):
    if st.session_state.img2 is not None:
        res = np.clip(ensure_gray(st.session_state.original_image).astype(int) + ensure_gray(st.session_state.img2).astype(int), 0, 255)
        st.session_state.processed_image = res.astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Img1", "Sum")

if st.button("Subtract Two Images"):
    if st.session_state.img2 is not None:
        res = np.clip(ensure_gray(st.session_state.original_image).astype(int) - ensure_gray(st.session_state.img2).astype(int), 0, 255)
        st.session_state.processed_image = res.astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Img1", "Diff")

# Histogram Operations
st.header("Histogram Operations")
if st.button("Show Histogram"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        show_histogram(img)

if st.button("Contrast Stretching"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        mn, mx = img.min(), img.max()
        res = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        st.session_state.processed_image = res
        display_images(st.session_state.original_image, res, "Original", "Contrast Stretched")

if st.button("Histogram Equalization"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        hist = compute_hist(img)
        cdf = np.zeros(256)
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]
        cdf_norm = np.round((cdf / img.size) * 255).astype(np.uint8)
        h, w = img.shape
        eq_img = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                eq_img[i, j] = cdf_norm[img[i, j]]
        st.session_state.processed_image = eq_img
        display_images(st.session_state.original_image, eq_img, "Original", "Equalized")

# Linear Filters
st.header("Linear Filters")
filter_type = st.selectbox("Filter Type", ["mean", "gaussian", "laplacian"], key="linear_filter")
if st.button("Apply Linear Filter"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        if filter_type == "mean":
            kernel = np.ones((3, 3)) / 9
        elif filter_type == "gaussian":
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        elif filter_type == "laplacian":
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        h, w = img.shape
        padded = np.pad(img, 1, mode='constant')
        out = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                out[i, j] = np.sum(region * kernel)
        st.session_state.processed_image = np.clip(out, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", f"{filter_type} Filter")

# Non-Linear Filters
st.header("Non-Linear Filters")
nl_filter_type = st.selectbox("Filter Type", ["median", "min", "max", "range", "mode"], key="nl_filter")
if st.button("Apply Non-Linear Filter"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        h, w = img.shape
        padded = np.pad(img, 1, mode='edge')
        out = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3].flatten()
                if nl_filter_type == "median":
                    val = np.median(region)
                elif nl_filter_type == "min":
                    val = np.min(region)
                elif nl_filter_type == "max":
                    val = np.max(region)
                elif nl_filter_type == "range":
                    val = np.max(region) - np.min(region)
                elif nl_filter_type == "mode":
                    counts = np.bincount(region)
                    val = np.argmax(counts)
                out[i, j] = val
        st.session_state.processed_image = out
        display_images(st.session_state.original_image, out, "Original", f"{nl_filter_type} Filter")

# Add Noise
st.header("Add Noise")
if st.button("Salt & Pepper Noise"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).copy()
        h, w = img.shape
        prob = 0.05
        for i in range(h):
            for j in range(w):
                r = random.random()
                if r < prob / 2:
                    img[i, j] = 0
                elif r < prob:
                    img[i, j] = 255
        st.session_state.processed_image = img
        display_images(st.session_state.original_image, img, "Original", "Salt & Pepper")

if st.button("Gaussian Noise"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        gauss = np.random.normal(0, 25, img.shape)
        noisy = img + gauss
        st.session_state.processed_image = np.clip(noisy, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", "Gaussian Noise")

if st.button("Periodic Noise"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        h, w = img.shape
        noisy = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                noise = 50 * math.sin(i / 2)
                noisy[i, j] = img[i, j] + noise
        st.session_state.processed_image = np.clip(noisy, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", "Periodic Noise")

# Morphology
st.header("Morphology")
morph_op = st.selectbox("Morphology Operation", ["dilation", "erosion", "opening", "closing"], key="morph")
if st.button("Apply Morphology"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        h, w = img.shape
        padded = np.pad(img, 1, mode='edge')
        out = np.zeros((h, w), dtype=np.uint8)
        if morph_op in ["dilation", "erosion"]:
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+3, j:j+3]
                    if morph_op == "dilation":
                        out[i, j] = np.max(region)
                    elif morph_op == "erosion":
                        out[i, j] = np.min(region)
            st.session_state.processed_image = out
        elif morph_op == "opening":
            # Erosion then Dilation
            eroded = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+3, j:j+3]
                    eroded[i, j] = np.min(region)
            padded_eroded = np.pad(eroded, 1, mode='edge')
            for i in range(h):
                for j in range(w):
                    region = padded_eroded[i:i+3, j:j+3]
                    out[i, j] = np.max(region)
            st.session_state.processed_image = out
        elif morph_op == "closing":
            # Dilation then Erosion
            dilated = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+3, j:j+3]
                    dilated[i, j] = np.max(region)
            padded_dilated = np.pad(dilated, 1, mode='edge')
            for i in range(h):
                for j in range(w):
                    region = padded_dilated[i:i+3, j:j+3]
                    out[i, j] = np.min(region)
            st.session_state.processed_image = out
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", morph_op.title())

# Segmentation & Dithering
st.header("Segmentation & Dithering")
if st.button("Auto Thresholding"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        T = 128
        while True:
            g1 = img[img > T]
            g2 = img[img <= T]
            if len(g1) == 0 or len(g2) == 0:
                break
            new_T = (np.mean(g1) + np.mean(g2)) / 2
            if abs(T - new_T) < 0.5:
                break
            T = new_T
        binary = np.zeros_like(img)
        binary[img > T] = 255
        st.session_state.processed_image = binary
        display_images(st.session_state.original_image, binary, "Original", f"Auto Threshold T={int(T)}")

if st.button("Floyd-Steinberg Dithering"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x]
                new_pixel = 255 if old_pixel > 128 else 0
                img[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                if x + 1 < w:
                    img[y, x + 1] += quant_error * 7 / 16
                if x - 1 >= 0 and y + 1 < h:
                    img[y + 1, x - 1] += quant_error * 3 / 16
                if y + 1 < h:
                    img[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w and y + 1 < h:
                    img[y + 1, x + 1] += quant_error * 1 / 16
        st.session_state.processed_image = np.clip(img, 0, 255).astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Original", "Floyd-Steinberg Dithering")








