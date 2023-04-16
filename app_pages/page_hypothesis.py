import streamlit as st


def page_hypothesis_body():
    st.write("### Hypotesis 1 and validation")

    st.success(
        f"Cherry leaves infected with powdery mildew have unique visual "
        f"characteristics that can be differentiated from healthy cherry "
        f"leaves using machine learning algorithms. We expect cherry leaves "
        f"infected with powdery mildew have white or grayish powdery coating "
        f"on the surface, typical symptom is white or grayish marks on the "
        f"leaves."
    )

    st.info(
        f"The model was able to identify the differences between data points "
        f"and learn how to accurately predict outcomes based on those "
        f"distinctions. A model has the ability to predict results on new "
        f"data without becoming too reliant on the training set. By doing so, "
        f"the model can generalize its predictions and make reliable forecasts "
        f"for future observations. Rather than simply memorizing the "
        f"relationships between features and labels in the training data, the "
        f"model learns to recognize general patterns, which enhances its "
        f"predictive power. The model v2 is able to do it with accuracy 99%."
    )

    st.write(
        f"To study detected visual features of infected and healthy leaves "
        f"visit the **Cherry Leaf Visualiser** tab.")


    st.write("### Hypotesis 2 and validation")

    st.success(
        f"An ML system trained on cherry leaf images can accurately "
        f"differentiate between healthy and powdery mildew-infected cherry "
        f"leaves with at least 90% accuracy. Based on the low complexity of "
        f"the binary classification task, it is reasonable to expect about 90% "
        f"accuracy rates."
    )

    st.info(
        f"The hypothesis has been verified by creating a model which has shown "
        f"excellent performance on the two main evaluation metrics for "
        f"assessing the business functionality and achievement of the project, "
        f"namely the overall F1 score 99% and the recall on the powdery "
        f"mildew label 99%."
    )
    
    st.write(
        f"To study the model perforamce metrics visit the **ML Prediction "
        f"Metrics** tab.")

    st.write("### Hypotesis 3 and validation")

    st.success(
        f"Implementing image visualisation in the cherry leaf inspection "
        f"process will result in a reduction of misidentification of infected "
        f"leaves"
    )

    st.info(
        f"Despite the ML model's high accuracy in predicting, the visual "
        f"features of the disease are hardly distinguishable during the "
        f"initial stages. Therefore, to reinforce the model and achieve a "
        f"higher final accuracy, visual examination of cases where the model "
        f"is not confident is carried out by a staff member during the ML "
        f"prediction process. This is accomplished using an image and a "
        f"prediction plot in the Powdery Mildew Detection tab."
    )

    st.write(
        f"To study the model performance metrics visit the "
        f"**Powdery Mildew Detection** tab.")

    st.write("### Hypotesis 4 and validation")

    st.success(
        f"The implementation of the ML solution can improve the accuracy and "
        f"speed of the cherry leaf inspection process, leading to more efficient use of "
        f"resources and increased productivity and increased worker safety by "
        f"reducing the amount of time and exposure required for manual "
        F"inspection of cherry leaves."
    )

    st.info(
        f"The business case states: 'An employee spends around 30 minutes in "
        f"each tree, taking a few samples of tree leaves and verifying "
        f"visually if the leaf tree is healthy or has powdery mildew. If it "
        f"has powdery mildew, the employee applies a specific compound to kill "
        f"the fungus. The time spent applying this compound is 1 minute.'\n\n"
        f"This would mean that it would take 50 hours to inspect and treat 100 "
        f"trees manually. Now, let's assume that taking pictures of the tree "
        f"leaves using a smartphone camera and uploading them to the ML model "
        f"for analysis takes approximately 1 minute per tree. This would mean "
        f"that it would take 1 hour and 40 minutes to take pictures of all 100 "
        f"trees and upload them for analysis.\n"
        f"The ML model has 99% accuracy in predicting the health of the trees, "
        f"the staff member could then focus their attention on the trees that "
        f"the model has identified as having powdery mildew. This would "
        f"significantly reduce the time and effort required for manual "
        f"verification, as the staff member would only need to visually verify "
        f"a subset of trees rather than all of them.\n"
        f"Overall, with the implementation of an ML model, the time required "
        f"for inspecting and treating 100 trees could be reduced from 50 hours "
        f"to approximately 2 hours, a savings of 48 hours or 96 % reduction in "
        f"time."
    )

    st.write("### Hypotesis 5 and validation")

    st.success(
        f"The use of the ML predicting can significantly reduce the "
        f"company's reliance on manual labour for detecting powdery mildew on "
        f"cherry leaves, resulting in cost savings and increased efficiency."
    )

    st.info(
        f"We will need to measure the reduction in manual labour needed for "
        f"detecting powdery mildew on cherry leaves after implementing the ML "
        f"solution. We can compare the time taken and cost incurred in the "
        f"manual inspection process before and after implementing the ML "
        f"system. Additionally, we can survey the employees involved in the "
        f"inspection process to gather their feedback on the effectiveness of "
        f"the ML system in reducing their workload and increasing their "
        f"efficiency."
    )
