# 🏭 Smart Factory AI — Predictive Maintenance

AI-based machine failure-risk prediction using sensor readings.

## Features
- Random Forest failure-risk prediction
- Temperature, vibration, pressure, RPM and torque inputs
- Failure risk, health score and machine status
- Prediction history
- CSV export
- Risk and health trend charts

## Tech Stack
Python, Streamlit, NumPy, Pandas, Scikit-learn, Matplotlib.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Choose **Create app**.
5. Select your repository, branch (`main`) and file (`app.py`).
6. Deploy.

## Important limitation
The current demonstration model is trained on synthetic sensor data. For real industrial use, retrain and validate it using actual machine sensor and maintenance/failure records.
