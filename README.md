# RetinaScope

### A clearer view of diabetic retinopathy, one retinal image at a time.

RetinaScope is a Streamlit app that predicts a diabetic retinopathy grade from a retinal fundus image and visualizes image regions that influenced the prediction with Grad-CAM.

<p align="center">
  <img src="assets/dashboard.png" alt="RetinaScope dashboard showing the retinal image upload area" width="900">
</p>

## What it can do

- Accept retinal fundus images in JPG, JPEG, or PNG format.
- Predict one of five grades: **No DR**, **Mild**, **Moderate**, **Severe**, or **Proliferative DR**.
- Display the model's confidence for its predicted grade.
- Generate a Grad-CAM visualization to help interpret the prediction.

## Run it locally

Python 3.10 or newer is recommended.

```powershell
git clone https://github.com/varshitha1809/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

The app loads `models/final_efficientnet_b0.pth` by default. To use a model file in another location, set `MODEL_PATH` before launching:

```powershell
$env:MODEL_PATH = "C:\path\to\your\model.pth"
python -m streamlit run app/app.py
```

## Project map

```text
.
├── app/
│   └── app.py
├── assets/
│   └── dashboard.png
├── colored_images/                 # Retinal image dataset
├── GradCAM/
│   └── gradcam_all_classes.png
├── models/                          # EfficientNet-B0 checkpoints
├── train.csv
├── requirements.txt
└── README.md
```

## A note on medical use

This project is for educational and research use only. Its predictions and Grad-CAM visualizations are not a diagnosis, medical advice, or a substitute for evaluation by a qualified eye-care professional.
