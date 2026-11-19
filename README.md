# GeoChange — Satellite/Drone Change Detection

A beginner-friendly Streamlit + OpenCV capstone prototype aligned with the four modules in the project PPT.

## Features
- Upload before-event and after-event images (PNG/JPG/TIFF)
- Sample imagery button for first-time testing
- Resize images and attempt affine alignment using OpenCV ECC
- LAB color-difference change detection
- Thresholding, morphology and connected-component region extraction
- Changed-area percentage and Low / Medium / High screening
- Change mask, red overlay and difference heatmap
- Region table and downloadable CSV / TXT report

## Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open the extracted `satellite_drone_change_detection_app` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Check Python is installed:
   ```powershell
   python --version
   ```
5. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
6. Activate it:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```cmd
   .venv\Scripts\activate.bat
   ```
7. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
8. Start the application:
   ```powershell
   streamlit run app.py
   ```
9. A browser tab should open at `http://localhost:8501`. If not, copy that address into Chrome.

## First test
- Open the Dashboard tab.
- Click **Load sample images**.
- Review the detected changed area, severity, mask, overlay and heatmap.
- Then upload your own BEFORE and AFTER images of the same place.

## Deploy online with Streamlit Community Cloud

1. Create/sign in to GitHub.
2. Create a new public repository, e.g. `satellite-drone-change-detection`.
3. Upload `app.py`, `requirements.txt`, and `README.md` from this folder (do not upload `.venv`).
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
6. Click **Deploy**. Wait for dependencies to install.
7. Copy the generated `*.streamlit.app` URL and share it with your faculty/team.

## Important academic limitation
This is a functional prototype of image-based visual change screening. It uses classical computer-vision methods, not a trained deep-learning semantic segmentation model. The Low/Medium/High thresholds are configurable prototype thresholds, not scientifically validated disaster-severity standards. Image differences may also be caused by lighting, season, camera angle, shadows or misalignment. Verify any disaster assessment with ground truth and expert review.
