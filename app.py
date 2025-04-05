import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import requests
from io import BytesIO
import numpy as np
import utils
import network
import test_dataset

# -----------------------------
# Configuration
# -----------------------------
MODEL_URL = "https://github.com/<your-username>/<your-repo>/releases/download/v1.0/deepfillv2.pth"
MODEL_PATH = "models/deepfillv2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)

# -----------------------------
# Download Pretrained Model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download model. HTTP Status: {response.status_code}")

    opt = type('', (), {})()
    opt.in_channels = 4
    opt.out_channels = 3
    opt.latent_channels = 48
    opt.pad_type = 'zero'
    opt.activation = 'elu'
    opt.norm = 'none'
    opt.init_type = 'xavier'
    opt.init_gain = 0.02

    generator = utils.create_generator(opt)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.to(DEVICE).eval()
    return generator

# -----------------------------
# Preprocess Inputs
# -----------------------------
def preprocess(image, mask):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image)
    mask = transform(mask)[0].unsqueeze(0)
    return image.unsqueeze(0), mask.unsqueeze(0)

# -----------------------------
# Postprocess Output
# -----------------------------
def postprocess(tensor):
    img = tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DeepFillv2 Inpainting", layout="centered")
st.title("🎨 DeepFillv2 - Image Inpainting with Mask")
st.markdown("Upload an image and corresponding binary mask (same dimensions).")

uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
uploaded_mask = st.file_uploader("Upload Mask", type=["jpg", "png", "jpeg"])

if uploaded_img and uploaded_mask:
    image = Image.open(uploaded_img).convert("RGB")
    mask = Image.open(uploaded_mask).convert("L")

    st.markdown("### Input Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(mask, caption="Mask", use_column_width=True)

    if st.button("Run Inpainting 🎨"):
        with st.spinner("Running DeepFillv2 model..."):
            generator = load_model()
            img_tensor, mask_tensor = preprocess(image, mask)
            img_tensor = img_tensor.to(DEVICE)
            mask_tensor = mask_tensor.to(DEVICE)

            with torch.no_grad():
                _, output = generator(img_tensor, mask_tensor)
                inpainted = img_tensor * (1 - mask_tensor) + output * mask_tensor
                output_image = postprocess(inpainted)

        st.markdown("### Inpainting Output")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(mask, caption="Mask", use_column_width=True)
        with col3:
            st.image(output_image, caption="Inpainted", use_column_width=True)

        # Download
        buf = BytesIO()
        output_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Output Image",
            data=byte_im,
            file_name="inpainted_output.png",
            mime="image/png"
        )
