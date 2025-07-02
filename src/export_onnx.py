# ----------------------------
# 6. Convert to ONNX
# ----------------------------
# src/export_onnx.py
from skl2onnx import to_onnx
import numpy as np
import joblib

pipeline = joblib.load("models/churn_model.pkl")
onnx_model = to_onnx(pipeline, X[:1].astype(np.float32))
with open("models/churn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
