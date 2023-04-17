import streamlit as st
from PIL import Image


image = Image.open("readme_assets/plot.jpg")


def page_instructions_body():
    st.write("**Here are usage guidelines for your Streamlit dashboard:**")

    st.info(
        f"1. *Project Summary:* This page provides an overview of the project, "
        f"including the dataset used, the problem statement, and the approach "
        f"taken to solve the problem."
        )
    
    st.info(
        f"2. *Cherry Leaf Visualiser:* Here you can study a visual difference "
        f"between healthy and affected by powdery mildew cherry leaves, "
        f"specifically the visual markers for defects that define powdery "
        f"mildew infection.\n\n Although the model is very accurate, the image "
        f"quality and high contrast might affect a prediction, so when the "
        f"model is not certain about the prediction result, you can study the "
        f"features of healthy and infected leaves to back up the model and "
        f"provide higher overall accuracy: ML model + human supervision.\n\n"
        )
    
    st.success(
        f"**Step by step**\n\n"
        f"To study the features of healthy and infected leaves, use image "
        f"montage feature.\n\n"
        f"1. Tick the last checkbox - **Image Montage**.\n"
        f"2. Choose the label you want to study from the dropdown menu.\n"
        f"3. Click the **Create Montage** button."
    )
    
    st.info(
        f"3. *Powdery Mildew Detection:* This page allows to upload "
        f"an image of a cherry leaf and predict whether it has powdery mildew "
        f"disease or not. The user can upload an image by clicking on the "
        f"**Browse files** button, selecting an image from their local "
        f"machine, and clicking on the **Upload** button. The prediction "
        f"result will be displayed on the page along with the confidence "
        f"score. The user can also view the predicted probabilities of the "
        f"input image for each class."
        )
    
    st.success(
        f"**Step by step**\n\n"
        f"1. Open the Streamlit dashboard in a web browser.\n"
        f"2. Navigate to the *Powdery Mildew Detection* page by clicking on "
        f"the corresponding tab in the sidebar menu.\n"
        f"3. Click on the **Browse files** button, select an image or a batch "
        f"from the local machine, and click on the **Upload** button.\n"
        f"4. Predictions will appear below. Read the prediction result or "
        f"study the features and metrics as required. To read the prediction "
        f"result, the user should look for the predicted class and the "
        f"corresponding confidence score. For example, if the predicted class "
        f"is **powdery mildew** and the confidence score is 0.85, it means "
        f"that the model is 85 % confident that the input image has powdery "
        f"mildew disease.\n"
        f"5. If needed, you can scroll through the predictions and check "
        f"where the model was uncertain and make a correct decision yourself."
        f"6. Download a prediction report at the bottom of the page."
    )

    st.image(image,
            caption="Prediction result plot")

    st.info(
        f"4. *Project Hypothesis:* The page provides a high-level summary of "
        f"the ML project and its expected outcomes, which can be useful for "
        f"stakeholders, business owners, executives, and managers who are "
        f"responsible for making strategic decisions based on the outcomes "
        f"of the project. "
    )

    st.info(
        f"5. *ML Prediction Metrics:* This page provides the evaluation "
        f"metrics of the machine learning model used in the project. The user "
        f"can view the confusion matrix, precision, recall, and F1 score of "
        f"the model. The user can also study the metrics to understand the "
        f"performance of the model. It is primarily intended for technical "
        f"staff members who are responsible for building and refining the ML "
        f"model, but may also be relevant for other stakeholders who are "
        f"interested in understanding the technical performance of the project."
    )