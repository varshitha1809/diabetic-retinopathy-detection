
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        PROJECT_ROOT / "models" / "final_efficientnet_b0.pth"
    )
)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Load model
# =========================================================

@st.cache_resource
def load_model():

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 5)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# =========================================================
# Image preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)


# =========================================================
# Title
# =========================================================

st.title("Diabetic Retinopathy Detection and Grading")

st.write(
    "Upload a retinal fundus image to predict the severity "
    "of diabetic retinopathy."
)


# =========================================================
# File upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Fundus Image")

    st.image(
        image,
        caption="Input retinal image",
        width=400
    )

    if st.button("Predict DR Grade"):

        # Prepare image
        resized_image = image.resize((224, 224))

        rgb_image = (
            np.array(resized_image)
            .astype(np.float32) / 255.0
        )

        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # -----------------------------------------------
        # Prediction
        # -----------------------------------------------

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        prediction = CLASS_NAMES[predicted_class]

        # -----------------------------------------------
        # Display prediction
        # -----------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"Predicted Grade: {prediction}"
        )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        # -----------------------------------------------
        # Grad-CAM
        # -----------------------------------------------

        st.subheader("Grad-CAM Explanation")

        target_layers = [model.features[-1]]

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        targets = [
            ClassifierOutputTarget(predicted_class)
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        st.image(
            visualization,
            caption="Grad-CAM: regions influencing the prediction",
            width=600
        )

        st.info(
            "Grad-CAM highlights image regions that contributed "
            "to the model's prediction. It is intended for "
            "model interpretability and not as a clinical diagnosis."
        )
