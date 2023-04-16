import streamlit as st
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_leaf_visualiser import page_leaf_visualiser_body
from app_pages.page_powdery_mildew_detection import page_powdery_mildew_detection_body
from app_pages.page_hypothesis import page_hypothesis_body
from app_pages.page_ml_prediction import page_ml_prediction_metrics

# Create an instance of the app
app = MultiPage(app_name="Powdery Mildew Detector")

# Add app pages here
app.add_page("Project Summary", page_summary_body)
app.add_page("Cherry Leaf Visualiser", page_leaf_visualiser_body)
app.add_page("Powdery Mildew Detection", page_powdery_mildew_detection_body)
app.add_page("Project Hypothesis", page_hypothesis_body)
app.add_page("ML Prediction Metrics", page_ml_prediction_metrics)

app.run()
