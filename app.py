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
    if len(img.shape) == 3:
        return ((img[:, :, 0].astype(int) +
                 img[:, :, 1].astype(int) +
                 img[:, :, 2].astype(int)) // 3).astype(np.uint8)
    return img

def display_images(img1, img2=None, title1="Original", title2="Processed"):
    if img2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=title1, width='auto')
        with col2:
            st.image(img2, caption=title2, width='auto')
    else:
        st.image(img1, caption=title1, width='auto')

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
    img2 = transform.resize(
        img2,
        st.session_state.original_image.shape[:2],
        preserve_range=True
    ).astype(np.uint8)
    st.session_state.img2 = img2
    display_images(st.session_state.img2, title1="Second Image")

if st.button("RGB to Gray (Manual)"):
    if st.session_state.original_image is not None:
        st.session_state.processed_image = ensure_gray(st.session_state.original_image)
        display_images(
            st.session_state.original_image,
            st.session_state.processed_image,
            "RGB",
            "Gray"
        )

# Point Operations
st.header("Point Operations")
brightness_op = st.selectbox("Brightness Operation", ["add", "mult"])
brightness_val = st.number_input("Value", value=0.0)

if st.button("Apply Brightness"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        res = img + brightness_val if brightness_op == "add" else img * brightness_val
        st.session_state.processed_image = np.clip(res, 0, 255).astype(np.uint8)
        display_images(
            st.session_state.original_image,
            st.session_state.processed_image,
            "Original",
            f"Brightness ({brightness_op})"
        )

darkness_op = st.selectbox("Darkness Operation", ["sub", "div"])
darkness_val = st.number_input("Value", value=1.0)

if st.button("Apply Darkness"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image).astype(float)
        res = img - darkness_val if darkness_op == "sub" else np.where(darkness_val != 0, img / darkness_val, img)
        st.session_state.processed_image = np.clip(res, 0, 255).astype(np.uint8)
        display_images(
            st.session_state.original_image,
            st.session_state.processed_image,
            "Original",
            f"Darkness ({darkness_op})"
        )

if st.button("Inverse (Negative)"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        st.session_state.processed_image = 255 - img
        display_images(
            st.session_state.original_image,
            st.session_state.processed_image,
            "Original",
            "Negative"
        )

# Arithmetic Operations
st.header("Arithmetic Operations")

if st.button("Add Two Images"):
    if st.session_state.img2 is not None:
        res = np.clip(
            ensure_gray(st.session_state.original_image).astype(int) +
            ensure_gray(st.session_state.img2).astype(int),
            0, 255
        )
        st.session_state.processed_image = res.astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Img1", "Sum")

if st.button("Subtract Two Images"):
    if st.session_state.img2 is not None:
        res = np.clip(
            ensure_gray(st.session_state.original_image).astype(int) -
            ensure_gray(st.session_state.img2).astype(int),
            0, 255
        )
        st.session_state.processed_image = res.astype(np.uint8)
        display_images(st.session_state.original_image, st.session_state.processed_image, "Img1", "Diff")

# Histogram Operations
st.header("Histogram Operations")

if st.button("Show Histogram"):
    if st.session_state.original_image is not None:
        show_histogram(ensure_gray(st.session_state.original_image))

if st.button("Contrast Stretching"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        mn, mx = img.min(), img.max()
        res = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        display_images(st.session_state.original_image, res, "Original", "Contrast Stretched")

if st.button("Histogram Equalization"):
    if st.session_state.original_image is not None:
        img = ensure_gray(st.session_state.original_image)
        hist = compute_hist(img)
        cdf = np.cumsum(hist)
        cdf_norm = np.round((cdf / img.size) * 255).astype(np.uint8)
        eq_img = cdf_norm[img]
        display_images(st.session_state.original_image, eq_img, "Original", "Equalized")





