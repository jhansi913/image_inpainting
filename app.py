import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import util1

import network
import test_dataset
import os
import urllib.request  # ✅ Required for downloading model

# ----------------------------------------
#              Configurations
# ----------------------------------------
MODEL_URL = "https://github.com/jhansi913/image_inpainting/releases/download/v1.0/deepfillv2_WGAN.pth"
MODEL_NAME = "deepfillv2_WGAN.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self):
        self.in_channels = 4
        self.out_channels = 3
        self.latent_channels = 48
        self.pad_type = 'zero'
        self.activation = 'elu'
        self.norm = 'none'
        self.init_type = 'xavier'
        self.init_gain = 0.02

opt = Args()

# ----------------------------------------
#        Load Generator Model
# ----------------------------------------

def download_model():
    if not os.path.exists(MODEL_NAME):
        print("🔄 Downloading model from GitHub...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print("✅ Model downloaded.")
    else:
        print("📦 Model already exists locally.")

@st.cache_resource
def load_model():
    download_model()  # ✅ Ensure the model gets downloaded
    generator = util1.create_generator(opt)
    generator.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    generator.to(DEVICE).eval()
    return generator

# ----------------------------------------
#         Preprocessing Functions
# ----------------------------------------

def preprocess(image_pil, mask_pil):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image_pil.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]
    mask = transform(mask_pil.convert("L")).unsqueeze(0)      # [1, 1, H, W]
    return image.to(DEVICE), mask.to(DEVICE)

def postprocess(tensor):
    tensor = tensor.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

# ----------------------------------------
#              Streamlit UI
# ----------------------------------------

st.set_page_config(page_title="DeepFillv2 Inpainting", layout="centered")
st.title("🖼️ DeepFillv2 Image Inpainting")
st.markdown("Upload an image and a mask to remove unwanted regions.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
uploaded_mask = st.file_uploader("Upload Mask (white = masked)", type=["jpg", "jpeg", "png"])

if uploaded_image and uploaded_mask:
    image = Image.open(uploaded_image)
    mask = Image.open(uploaded_mask)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(mask, caption="Mask", use_column_width=True)

    if st.button("Run Inpainting 🎨"):
        with st.spinner("Running DeepFillv2..."):
            generator = load_model()
            img_tensor, mask_tensor = preprocess(image, mask)

            with torch.no_grad():
                first_out, second_out = generator(img_tensor, mask_tensor)
                result_tensor = img_tensor * (1 - mask_tensor) + second_out * mask_tensor
                result_image = postprocess(result_tensor)

        st.markdown("### ✨ Inpainting Output")
        st.image(result_image, use_column_width=True)

        # Download button
        st.download_button(
            label="📥 Download Inpainted Image",
            data=result_image.convert("RGB").tobytes(),
            file_name="inpainted_output.png",
            mime="image/png"
        )
