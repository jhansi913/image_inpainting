import streamlit as st
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
from utils import create_generator
from test_dataset import tensor_to_image
import yaml
import types

# Load options
class Options:
    def __init__(self):
        self.init_type = 'kaiming'
        self.init_gain = 0.02
        self.input_size = 256
        self.gpu_ids = []
        self.padding = 'SAME'
        self.rate = 2
        self.threshold = 0.7
        self.iteration = 1
        self.upsample = True

opt = Options()

# Load pretrained model
@st.cache_resource
def load_model():
    model = create_generator(opt)
    model.load_state_dict(torch.load('deepfillv2.pth', map_location='cpu'))
    model.eval()
    return model
def normalize(img_tensor):
    return (img_tensor - 0.5) / 0.5

 

def preprocess_image(image, mask, size=256):
    # Resize to input size and convert to tensor
    image = image.resize((size, size)).convert("RGB")
    mask = mask.resize((size, size)).convert("L")

    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    mask_tensor = transforms.ToTensor()(mask).unsqueeze(0)
    ones_tensor = torch.ones_like(mask_tensor)

    return normalize(img_tensor), mask_tensor, ones_tensor

def inpaint(model, image, mask, ones):
    with torch.no_grad():
        first_out, second_out, offset_flow = model(image, mask, ones)
        return second_out

# Streamlit UI
st.title("🖼️ DeepFillv2 Image Inpainting")

uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
uploaded_mask = st.file_uploader("Upload Mask", type=['jpg', 'png', 'jpeg'])

if uploaded_image and uploaded_mask:
    image = Image.open(uploaded_image)
    mask = Image.open(uploaded_mask)

    st.image([image, mask], caption=["Original Image", "Mask"], width=256)

    model = load_model()
    norm_img, mask_tensor, ones_tensor = preprocess_image(image, mask)

    output = inpaint(model, norm_img, mask_tensor, ones_tensor)
    result_image = tensor_to_image(output)

    st.subheader("🧠 Inpainted Result")
    st.image(result_image, caption="Output", width=256)

    # Convert to downloadable file
    result_pil = Image.fromarray(result_image)
    st.download_button(
        label="Download Result",
        data=cv2.imencode('.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="inpainting_result.png",
        mime="image/png"
    )
