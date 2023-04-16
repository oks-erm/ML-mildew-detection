import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


image = Image.open("readme_assets/summary.jpg")


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a fungal disease that typically affects cherry "
        f"trees. Manual verification of powdery mildew on cherry trees is "
        f"currently performed by visually inspecting the leaves, which is not "
        f"scalable due to the large number of cherry trees in multiple "
        f"farms.\n\n"
        f"The proposed solution is a machine learning (ML) system capable of "
        f"detecting instantly, using a cherry leaf image, whether it is "
        f"healthy or has powdery mildew.\n\n"
        f"**Project Dataset**\n\n"
        f"The dataset is a collection of cherry leaf images provided by "
        f"**Farmy & Foods**, taken from their crops."
    )

    st.image(image, 
             caption="Powdery Mildew Detection predicts on cherry leaves images",
             use_column_width=True)

    st.write(
        f"* For additional information on the dataset and data preparation, "
        f"see the [README file]"
        f"(https://github.com/oks-erm/ML-mildew-detection/blob/main/README.md).")

    st.success(
        f"**Business requirements:**\n"
        f"* 1 - The client is interested in conducting a study to visually "
        f"differentiate a healthy cherry leaf from one infected with powdery "
        f"mildew.\n"
        f"* 2 - The client is interested in quickly and accurately predicting "
        f"if a cherry tree is healthy or contains powdery mildew.\n"
        f"* 3 - The client wants a dashboard that meets the above "
        f"requirements.\n"
        f"* 4 - The expected outcomes should be explained.\n"
    )

    st.info(
        f"**Objectives**\n\n"
        f"* To develop a ML system that can accurately detect whether a "
        f"cherry leaf is healthy or infected with powdery mildew using an "
        f"image of the leaf.\n"
        f"* To improve the efficiency of powdery mildew detection on cherry "
        f"trees in multiple farms by providing an instant and scalable "
        f"solution.\n"
    )

    st.info(
        f"**Processes**\n\n"
        f"1. Collect a dataset of cherry leaf images provided by Farmy & "
        f"Foods.\n"
        f"2. Preprocess the dataset by cleaning, resizing, and normalizing the "
        f"images to prepare them for ML algorithms.\n"
        f"3. Image augmentation: augment the training dataset images to "
        f"increase the model's performance.\n"
        f"4. Develop a ML model using supervised learning techniques, such as "
        f"convolutional neural networks, to classify cherry leaves as healthy "
        f"or infected with powdery mildew.\n"
        f"5. Train the model using the preprocessed dataset and validate its "
        f"performance on a separate test dataset.\n"
        f"6. Deploy the trained model to a dashboard that displays its "
        f"predictions and meets the client's requirements.\n"
        f"7. Conduct a study to visually differentiate healthy and powdery "
        f"mildew-infected cherry leaves and predict the health of cherry trees "
        f"using the deployed model.\n"
        f"8. Provide possible outcomes and explanations to the client to "
        f"facilitate decision-making.\n"
    )
