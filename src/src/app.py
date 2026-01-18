# comments: Streamlit app to serve a trained Iris classifier.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from joblib import load


def load_artifacts() -> tuple[object, dict]:
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "decision_tree_classifier_default_42.joblib"
    metrics_path = base_dir / "models" / "metrics_default_42.json"

    model = load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model, metrics


def main() -> None:
    st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="centered")

    st.title("🌸 Iris - Predicción con Decision Tree")
    st.markdown("Ajusta los valores y ejecuta una predicción del tipo de Iris.")

    model, metrics = load_artifacts()
    feature_names = metrics["feature_names"]
    target_names = metrics["target_names"]

    st.sidebar.header("⚙️ Parámetros del modelo")
    st.sidebar.write(f"Accuracy (test): **{metrics['accuracy']:.4f}**")
    st.sidebar.write("random_state: **42**")

    st.subheader("🧪 Inputs")

    # comments: Use realistic Iris ranges with safe defaults.
    val1 = st.slider(feature_names[0], min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    val2 = st.slider(feature_names[1], min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    val3 = st.slider(feature_names[2], min_value=1.0, max_value=7.0, value=4.0, step=0.1)
    val4 = st.slider(feature_names[3], min_value=0.1, max_value=2.6, value=1.3, step=0.1)

    X = np.array([[val1, val2, val3, val4]], dtype=float)

    if st.button("🔮 Predict"):
        pred = int(model.predict(X)[0])
        st.success(f"Predicción: **{target_names[pred]}**")

        proba = getattr(model, "predict_proba", None)
        if callable(proba):
            probs = proba(X)[0]
            st.write("Probabilidades:")
            st.json({target_names[i]: float(probs[i]) for i in range(len(target_names))})

    st.divider()
    st.caption("Deploy en Render: streamlit run app.py --server.port $PORT --server.address 0.0.0.0")


if __name__ == "__main__":
    main()
