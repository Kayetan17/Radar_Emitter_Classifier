

import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from PulseCreator import EMITTERS, generate_pulse_offsets, generate_adaptive_train
from features import extract_features

st.set_page_config(page_title="Radar Emitter Classifier", layout="centered")


@st.cache_resource
def load_models():
    bundle = joblib.load("model.joblib")
    return bundle["naive_model"], bundle["robust_model"], bundle["radar_names"]


naive_model, robust_model, radar_names = load_models()

st.title("Radar Emitter Classifier")
st.write(
    "Pick a radar and transmit a burst of pulses. The classifier will try to identify "
    "which radar it is. Turn on **cognitive mode** to make the radar shift its "
    "fingerprint mid burst, then watch the Standard Classifier get fooled, and the "
    "Counter Cognitive Classifier see through it."
)

col1, col2 = st.columns(2)

with col1:
    chosen_radar = st.selectbox("Radar", radar_names)
    cognitive = st.checkbox("Cognitive mode")

with col2:
    model_choice = st.radio("Model", ["Standard Classifier", "Counter Cognitive Classifier"])
    n_pulses = st.slider("Pulses received", 6, 20, 8)

if st.button("Transmit", type="primary"):
    params = EMITTERS[chosen_radar]
    pri, pw, freq = params["pri"], params["pw"], params["frequency"]

    if cognitive:
        train = generate_adaptive_train(pri, pw, freq, n_pulses=n_pulses)
    else:
        train = generate_pulse_offsets(pri, pw, freq, n_pulses)

    feats = extract_features(train).reshape(1, -1)
    model = naive_model if model_choice == "Standard Classifier" else robust_model

    probs = model.predict_proba(feats)[0]
    pred_index = int(np.argmax(probs))
    pred_name = radar_names[pred_index]
    confidence = probs[pred_index] * 100
    correct = (pred_name == chosen_radar)

    if correct:
        st.success(f"Identified as **{pred_name}** ({confidence:.0f}% confident) — correct")
    else:
        st.error(
            f"Identified as **{pred_name}** ({confidence:.0f}% confident) — "
            f"WRONG (true radar was {chosen_radar})"
        )
        if cognitive and model_choice == "Standard Classifier":
            st.info(
                "This is the cognitive radar problem: the radar changed its "
                "fingerprint mid burst, so the Standard Classifier, trained only on "
                "steady radars, misidentifies it. Try the Counter Cognitive Classifier on the "
                "same signal."
            )

    fig, ax = plt.subplots(figsize=(6, 3))
    bar_colors = ["#2c7fb8" if n == chosen_radar else "#cccccc" for n in radar_names]
    ax.bar(radar_names, probs * 100, color=bar_colors)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model confidence across radars (blue = true radar)")
    st.pyplot(fig)

    with st.expander("See the raw pulse train"):
        st.write("Each row is one pulse: [PRI (µs), PW (µs), Frequency (GHz)]")
        st.write("Notice: in cognitive mode the PRI jumps partway down the list.")
        st.dataframe(
            {
                "PRI (µs)": np.round(train[:, 0], 1),
                "PW (µs)": np.round(train[:, 1], 2),
                "Freq (GHz)": np.round(train[:, 2], 2),
            }
        )

st.divider()
st.caption(
    "Random Forest classifier on parameter level pulse features. "
    "Standard Classifier trained on steady radars only. The Counter Cognitive Classifier trained on steady "
    "+ adaptive radars with a drift feature."
)