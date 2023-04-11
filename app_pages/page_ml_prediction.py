import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.ml.evaluate_clf import load_test_evaluation


def page_ml_prediction_metrics(version='v1'):

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    # Displaying labels distribution on train, validation and test sets
    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')
    st.write(f"From these metrics, we can see that the train, validation and "
             f"test sets are well balanced in terms of class distribution "
             f"between healthy and powdery_mildew images, as they each have an "
             f"equal number of images for each class. This is important to "
             f"ensure that the model learns to distinguish between the two "
             f"classes equally well and avoids bias towards one class. ")
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        # Displaying model training accuracy
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        # Displaying model training losses
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.write("---")

    st.write("### Generalised Performance on Test Set")

    # Loading and displaying test set evaluation metrics (loss and accuracy)
    evaluation = load_test_evaluation(version)
    df_evaluation = pd.DataFrame(evaluation, index=['Loss', 'Accuracy'])
    st.dataframe(df_evaluation)

    st.write("---")

    # Displaying the confusion matrix
    confusion_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption="Confusion Matrix", width=500)

    st.write(f"The matrix has four quadrants: true positives (TP), true "
             f"negatives (TN), false positives (FP), and "
             f"false negatives (FN). TP and TN indicate correct predictions, "
             f"while FP and FN indicate incorrect predictions. The accuracy "
             f"metric tells us the percentage of correct predictions made by "
             f"the model - {100*evaluation[1]:.2f}%, while the loss metric - "
             f"{100*evaluation[0]:.2f}% - measures the deviation between the "
             f"predicted and true labels. A high accuracy and low loss indicate "
             f"that the model is making accurate predictions.")
    st.write("---")

    st.write("### Classification Report")

    # Displaying the classification report
    classification_report = plt.imread(
        f"outputs/{version}/classification_report.png")
    st.image(classification_report, caption='Classification Report')

    st.write(f"The classification report shows that the model has a high "
             f"precision and recall for both classes. **Precision** is the "
             f"number of true positives divided by the sum of true positives "
             f"and false positives. **Recall** is the number of true positives "
             f"divided by the sum of true positives and false negatives. "
             f"High precision indicates that the model can correctly identify "
             f"true positives with a low rate of false positives. High recall "
             f"indicates that the model can correctly identify true positives "
             f"with a low rate of false negatives.")

    st.write("---")

    st.write("### ROC Curve")

    # Displaying the ROC curve
    roc_curve = plt.imread(f"outputs/{version}/roc_curve.png")
    st.image(roc_curve, caption='ROC Curve')

    st.write(f"The ROC curve shows that the model has a high true positive rate "
             f"(sensitivity) and low false positive rate (1 - specificity) for "
             f"a range of thresholds. This indicates that the model has a high "
             f"ability to correctly classify positive samples as positive and "
             f"negative samples as negative. The AUC score of the ROC curve is "
             f"0.997, which is close to 1 and indicates that the model has "
             f"excellent performance in distinguishing between the 'Healthy' "
             f"and 'Powdery Mildew' classes.")
