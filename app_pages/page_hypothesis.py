import streamlit as st
import matplotlib.pyplot as plt


def page_hypothesis_body():
    st.write("### Hypotesis 1 and validation")

    st.success(
        f"Infected leaves have distinctive visual features differentiating "
        f"them from healthy leaves."
    )
    st.info(
        f"We expect cherry leaves infected with powdery mildew have white "
        f"or grayish powdery coating on the surface, typical symptom is "
        f"white or grayish marks on the leaves. These marks can sometimes "
        f"appear as irregular blotches or spots on the surface of the "
        f"leaf.\n\n"
    )
    st.write(
        f"To study detected visual features of infected and healthy leaves "
        f"visit the [Cherry Leaf Visualiser](#cherry-leaf-visualiser).")

    st.warning(
        f"The model was able to identify the differences between data points "
        f"and learn how to accurately predict outcomes based on those "
        f"distinctions. A model has the ability to predict results on new "
        f"data without becoming too reliant on the training set. By doing so, "
        f"the model can generalize its predictions and make reliable forecasts "
        f"for future observations. Rather than simply memorizing the "
        f"relationships between features and labels in the training data, the "
        f"model learns to recognize general patterns, which enhances its predictive power."
    )

    st.write("### Hypotesis 2 and validation")

    st.success(
        f""
    )
    st.info(
        f""
    )
    