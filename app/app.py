import os
import io
import html
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    CAM_AVAILABLE = True
except ImportError:
    CAM_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retinal Image Review",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        PROJECT_ROOT / "models" / "final_efficientnet_b0.pth"
    )
)

CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

SHORT_CLASSES = [
    "No DR", "Mild", "Moderate", "Severe", "Proliferative"
]

DESCRIPTIONS = [
    "The model classified this image as showing no diabetic retinopathy.",
    "The model predicts a pattern associated with mild diabetic retinopathy.",
    "The model predicts a pattern associated with moderate diabetic retinopathy.",
    "The model predicts a pattern associated with severe diabetic retinopathy.",
    "The model predicts a pattern associated with proliferative diabetic retinopathy."
]

COLORS = [
    "#39725F",
    "#718C51",
    "#AD813D",
    "#B96D4A",
    "#A95663"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown("""
<style>
:root {
    --bg: #202623;
    --panel: #2A312D;
    --mint: #91AD9B;
    --muted: #B5BFB8;
    --white: #E9EEEA;
    --line: #424B45;
}

.stApp {
    background: var(--bg);
    color: var(--white);
    font-family: "Segoe UI", Arial, sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1380px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    font-family: "Segoe UI", Arial, sans-serif;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border-radius: 12px;
    border: 1px solid var(--line);
    border-left: 4px solid #718C79;
    background: #292F2C;
    box-shadow: none;
    margin-bottom: 24px;
}

.eyebrow {
    color: var(--mint);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero-title {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: clamp(30px, 3.5vw, 42px);
    font-weight: 650;
    line-height: 1.2;
    letter-spacing: -0.6px;
    color: #E9EEEA;
    margin: 0;
}

.hero-title span {
    color: #A4BDAA;
}

.hero-sub {
    color: #B5BFB8;
    font-size: 14px;
    line-height: 1.7;
    max-width: 760px;
    margin-top: 12px;
}

.tag {
    display: inline-block;
    margin: 14px 7px 0 0;
    padding: 6px 9px;
    border-radius: 5px;
    border: 1px solid #48544C;
    background: #303834;
    color: #C4D0C7;
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.2px;
}

.live-status {
    margin-top: 14px;
    font-size: 11px;
    color: #AEB8B1;
}

.live-status span {
    color: #9FB7A6;
}

.section-title {
    font-family: "Segoe UI", Arial, sans-serif;
    color: #E4EAE5;
    font-size: 21px;
    font-weight: 650;
    letter-spacing: -0.2px;
    margin: 16px 0 5px;
}

.section-sub {
    color: #B5BFB8;
    font-size: 12px;
    line-height: 1.8;
    margin-bottom: 18px;
}

.panel {
    background: #2A312D;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 21px;
    margin-bottom: 17px;
    box-shadow: none;
}

.kicker {
    color: var(--mint);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.panel-title {
    color: #E2E9E3;
    font-size: 15px;
    font-weight: 650;
    margin-bottom: 7px;
}

.panel-copy {
    color: #B5BFB8;
    font-size: 12px;
    line-height: 1.8;
}

div[data-testid="stFileUploader"] {
    background: #252C28;
    border: 1px dashed #5A685F;
    border-radius: 10px;
    padding: 12px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
    color: #D5DDD7;
}

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 46px;
    border: 1px solid #657D6D;
    border-radius: 8px;
    color: #F3F6F3;
    background: #536B5B;
    font-size: 13px;
    font-weight: 650;
    transition: background-color 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    color: #FFFFFF;
    border: 1px solid #6B8574;
    background: #617967;
    box-shadow: none;
}

.grade-tile {
    min-height: 86px;
    padding: 13px 5px;
    border-radius: 9px;
    border: 1px solid var(--line);
    background: #2A312D;
    text-align: center;
}

.grade-number {
    color: #AAB5AE;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
}

.grade-bar {
    height: 4px;
    border-radius: 8px;
    margin: 11px 0;
}

.grade-name {
    color: #DCE4DE;
    font-size: 10px;
    font-weight: 650;
}

.metric-card {
    padding: 19px;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #2A312D;
    min-height: 112px;
}

.metric-label {
    color: #B5BFB8;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.metric-value {
    font-family: "Segoe UI", Arial, sans-serif;
    color: #E9EEEA;
    font-size: 25px;
    font-weight: 650;
    margin-top: 9px;
}

.metric-note {
    color: #AAB5AE;
    font-size: 10px;
    margin-top: 5px;
    line-height: 1.6;
}

.result-card {
    border: 1px solid var(--line);
    border-left: 4px solid var(--mint);
    border-radius: 10px;
    padding: 22px;
    background: #2A312D;
    margin: 10px 0 18px;
}

.result-label {
    color: #B5BFB8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 10px;
    font-weight: 800;
}

.result-grade {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: clamp(28px, 3vw, 39px);
    font-weight: 650;
    margin: 11px 0;
    line-height: 1.2;
}

.result-copy {
    color: #C1CAC3;
    font-size: 12px;
    line-height: 1.9;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    color: #D0D8D2;
    font-size: 12px;
    margin: 14px 0 7px;
}

div[data-testid="stProgressBar"] > div {
    background: #3B443F;
    border-radius: 100px;
}

div[data-testid="stProgressBar"] > div > div {
    background: #718C79;
    border-radius: 100px;
}

div[data-testid="stImage"] img {
    border-radius: 8px;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
}

hr {
    border-color: var(--line);
}

.footer {
    text-align: center;
    color: #99A49C;
    font-size: 10px;
    line-height: 2;
    padding-top: 28px;
    letter-spacing: 0.3px;
}

@media (max-width: 700px) {
    .hero { padding: 25px 21px; }
    .panel { padding: 16px; }
    .hero-title { letter-spacing: -1px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}\n"
            "Check that the model file exists or set the MODEL_PATH environment variable."
        )

    model = efficientnet_b0(weights=None)

    features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(features, 5)
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported model checkpoint format.")

    checkpoint = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, tensor, predicted_class, rgb_image):

    if not CAM_AVAILABLE:
        return None

    try:
        target_layers = [model.features[-1]]

        with GradCAM(
            model=model,
            target_layers=target_layers
        ) as cam:

            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[
                    ClassifierOutputTarget(int(predicted_class))
                ]
            )[0]

        return show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

    except Exception as error:
        st.warning(f"Grad-CAM generation failed: {error}")
        return None


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">RETINAL IMAGE REVIEW</div>
    <div class="hero-title">
        Diabetic retinopathy<br>
        <span>grading dashboard</span>
    </div>
    <div class="hero-sub">
        A research dashboard for reviewing a retinal image, model-generated
        grade estimates, and visual explanations.
    </div>
    <span class="tag">FIVE-GRADE CLASSIFICATION</span>
    <span class="tag">IMAGE EXPLANATION</span>
    <div class="live-status">
        For research and education only · Not for clinical diagnosis
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# WORKSPACE
# ============================================================

st.markdown(
    '<div class="section-title">Analysis workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-sub">'
    'Upload a retinal fundus photograph to inspect the model prediction '
    'and its visual explanation.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([0.96, 1.04], gap="large")


with left:

    st.markdown("""
    <div class="panel">
        <div class="kicker">01 / IMAGE INPUT</div>
        <div class="panel-title">Retinal fundus photograph</div>
        <div class="panel-copy">
            Choose a clear PNG, JPG or JPEG retinal image.
            For meaningful results, use a retinal fundus photograph.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded retinal image",
                use_container_width=True
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height} pixels"
            )

        except Exception:
            st.error("Unable to read this image file.")

    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:38px 18px;">
            <div style="font-size:42px;color:#75EBD0;margin-bottom:12px;">
                ◎
            </div>
            <div class="panel-title">Ready for image input</div>
            <div class="panel-copy">
                Your selected image will appear here before analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    analyze = st.button(
        "✦  Analyze retinal image",
        use_container_width=True
    )


with right:

    st.markdown("""
    <div class="panel">
        <div class="kicker">02 / SEVERITY REFERENCE</div>
        <div class="panel-title">Five-grade classification scale</div>
        <div class="panel-copy">
            The model predicts one of the five categories used during training.
        </div>
    </div>
    """, unsafe_allow_html=True)

    grade_columns = st.columns(5)

    for i, col in enumerate(grade_columns):

        with col:

            st.markdown(
                f"""
                <div class="grade-tile">
                    <div class="grade-number">GRADE {i}</div>
                    <div class="grade-bar"
                         style="background:{COLORS[i]};"></div>
                    <div class="grade-name">{SHORT_CLASSES[i]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="panel" style="margin-top:17px;">
        <div class="kicker">03 / REPORT CONTENTS</div>
        <div class="panel-title">Your analysis includes</div>
        <div class="panel-copy">
            <p>01 &nbsp; Predicted grade and confidence</p>
            <p>02 &nbsp; Probability profile across five classes</p>
            <p>03 &nbsp; Original image and Grad-CAM overlay</p>
            <p>04 &nbsp; Downloadable visualization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION AND RESULTS
# ============================================================

if analyze:

    if image is None:

        st.warning("Please upload a valid retinal fundus image first.")

    else:

        try:

            with st.spinner("Loading model and analyzing the image..."):

                model = load_model()

                resized = image.resize(
                    (224, 224),
                    Image.Resampling.BILINEAR
                )

                rgb_image = (
                    np.asarray(resized).astype(np.float32) / 255.0
                )

                input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():

                    logits = model(input_tensor)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )[0].cpu().numpy()

                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])

                cam_image = generate_gradcam(
                    model,
                    input_tensor,
                    predicted_class,
                    rgb_image
                )

            st.markdown("---")

            st.markdown(
                '<div class="section-title">Analysis report</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-sub">'
                'Model-generated results for the uploaded image. '
                'Confidence is not the same as clinical certainty.'
                '</div>',
                unsafe_allow_html=True
            )

            # Summary metric cards
            metric1, metric2, metric3 = st.columns(3, gap="medium")

            with metric1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted grade</div>
                        <div class="metric-value"
                             style="font-size:20px;color:{COLORS[predicted_class]}">
                            {html.escape(CLASSES[predicted_class])}
                        </div>
                        <div class="metric-note">Highest-scoring class</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Model confidence</div>
                        <div class="metric-value">{confidence * 100:.1f}%</div>
                        <div class="metric-note">
                            Softmax probability for the predicted class
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric3:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Inference device</div>
                        <div class="metric-value" style="font-size:21px">
                            {"GPU" if DEVICE.type == "cuda" else "CPU"}
                        </div>
                        <div class="metric-note">
                            Hardware used for inference
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Main prediction card
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        MODEL INTERPRETATION · GRADE {predicted_class}
                    </div>
                    <div class="result-grade"
                         style="color:{COLORS[predicted_class]}">
                        {html.escape(CLASSES[predicted_class])}
                    </div>
                    <div class="result-copy">
                        {html.escape(DESCRIPTIONS[predicted_class])}
                        <br><br>
                        This is an educational model prediction and is not
                        a confirmed clinical diagnosis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            probability_column, explanation_column = st.columns(
                [0.92, 1.08],
                gap="large"
            )

            # Probability chart
            with probability_column:

                st.markdown(
                    '<div class="section-title" style="font-size:18px;">'
                    'Class probability profile</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-sub">'
                    'Relative model scores across all five categories.'
                    '</div>',
                    unsafe_allow_html=True
                )

                for i, class_name in enumerate(CLASSES):

                    st.markdown(
                        f"""
                        <div class="prob-row">
                            <span>{html.escape(class_name)}</span>
                            <span style="color:{COLORS[i]};font-weight:800">
                                {probabilities[i] * 100:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(float(probabilities[i]))

            # Original and Grad-CAM
            with explanation_column:

                st.markdown(
                    '<div class="section-title" style="font-size:18px;">'
                    'Visual explanation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-sub">'
                    'Grad-CAM highlights image regions that influenced '
                    'the model prediction. It does not confirm a lesion.'
                    '</div>',
                    unsafe_allow_html=True
                )

                original_column, cam_column = st.columns(2, gap="medium")

                with original_column:

                    st.markdown(
                        '<div class="panel-title">Original image</div>',
                        unsafe_allow_html=True
                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                with cam_column:

                    st.markdown(
                        '<div class="panel-title">Grad-CAM overlay</div>',
                        unsafe_allow_html=True
                    )

                    if cam_image is not None:

                        st.image(
                            cam_image,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Grad-CAM is unavailable. Install "
                            "pytorch-grad-cam to enable the heatmap."
                        )

                if cam_image is not None:

                    buffer = io.BytesIO()

                    Image.fromarray(cam_image).save(
                        buffer,
                        format="PNG"
                    )

                    st.download_button(
                        "↓  Download Grad-CAM overlay",
                        data=buffer.getvalue(),
                        file_name="retinaai_gradcam_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )

            st.warning(
                "ACADEMIC RESEARCH PROTOTYPE ONLY. This model has not "
                "been validated for clinical use. Do not use its output "
                "to diagnose, rule out, or make treatment decisions about "
                "diabetic retinopathy. Consult a qualified eye-care professional."
            )

        except Exception as error:

            st.error(f"Analysis failed: {error}")

            st.caption(
                "Check that the model checkpoint exists and matches the "
                "EfficientNet-B0 architecture, "
                "and all required packages are installed."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    RETINAL IMAGE REVIEW · RESEARCH PROTOTYPE<br>
    Diabetic Retinopathy Detection &amp; Grading<br>
    EfficientNet-B0 · Five-class prediction · Grad-CAM explainability<br>
    Academic demonstration only · Not for clinical diagnosis
</div>
""", unsafe_allow_html=True)
