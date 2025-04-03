import streamlit as st
import torch
import os
import numpy as np
from PIL import Image
import utils  # Ensure this module contains create_generator and save_sample_png functions
import test_dataset
from torchvision import transforms

def load_generator():
    """Load the trained model."""
    model_path = os.path.join("saved_model", "deepfillv2_WGAN.pth")
    generator = utils.create_generator(opt)
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()
    return generator

def preprocess_image(image):
    """Convert image to tensor."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def main():
    st.title("Image Inpainting with DeepFillv2")
    
    st.sidebar.header("Upload Images")
    uploaded_image = st.sidebar.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
    uploaded_mask = st.sidebar.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image and uploaded_mask:
        image = Image.open(uploaded_image).convert("RGB")
        mask = Image.open(uploaded_mask).convert("L")
        
        st.image([image, mask], caption=["Original Image", "Mask Image"], width=300)
        
        if st.button("Generate Inpainted Image"):
            generator = load_generator()
            img_tensor = preprocess_image(image)
            mask_tensor = preprocess_image(mask)
            mask_tensor = torch.cat((mask_tensor, mask_tensor, mask_tensor), 1)  # Convert to 3 channels
            
            with torch.no_grad():
                _, output_img = generator(img_tensor, mask_tensor)
                output_img = output_img.squeeze().permute(1, 2, 0).numpy()
                output_img = (output_img * 255).astype(np.uint8)
                output_pil = Image.fromarray(output_img)
            
            st.image(output_pil, caption="Inpainted Image", use_column_width=True)

if __name__ == "__main__":
    main()
