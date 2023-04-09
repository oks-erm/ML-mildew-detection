import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from typing import Tuple, Union, Literal
from src.data_management import load_pkl_file


def plot_probabilities(pred_proba: float, pred_class: Union[Literal['Healthy'], Literal['Powdery Mildew']]):
    """
    Plot prediction probability
    """
    PROBABILITY_COLS = ['Probability', 'Diagnostic']
    CATEGORIES = {'Healthy': 0, 'Powdery Mildew': 1}
    NUM_CATEGORIES = len(CATEGORIES)

    prob_per_class = pd.DataFrame(
        data=[0] * NUM_CATEGORIES,
        index=CATEGORIES.keys(),
        columns=[PROBABILITY_COLS[0]]
    )

    prob_per_class.loc[pred_class] = pred_proba
    for cat in prob_per_class.index.to_list():
        if cat != pred_class:
            prob_per_class.loc[cat] = 1 - pred_proba

    prob_per_class = prob_per_class.round(3)
    prob_per_class[PROBABILITY_COLS[1]] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x=PROBABILITY_COLS[1],
        y=PROBABILITY_COLS[0],
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)


def resize_input_image(img: np.ndarray, version: str) -> np.ndarray:
    """
    Resize input image according to defined image shape for a specified version
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(
        input_image: np.ndarray, version: str) -> Tuple[float, str]:
    """
    Load and perform ML prediction on live images
    """
    CLASS_THRESHOLD = 0.5
    model = load_model(f"outputs/{version}/mildew_detector_model.h5")

    pred_proba = model.predict(input_image)[0, 0]

    target_map = {v: k for k, v in
                  {'Healthy': 0, 'Powdery Mildew': 1}.items()}
    pred_class = target_map[pred_proba > CLASS_THRESHOLD]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    if pred_class.lower() == 'healthy':
        st.markdown('<span style="background-color:#abf0d5; padding: 5px;">'
                    f"The sample cherry leave prediction is **{pred_class.lower()}**."
                    '</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span style="background-color:#fcb7b7; padding: 5px;">'
                    f"The sample cherry leave prediction is **{pred_class.lower()}**."
                    '</span>',
                    unsafe_allow_html=True)

    return pred_proba, pred_class
