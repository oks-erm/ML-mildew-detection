# Powdery Mildew Detector (ML project)

![mockup](readme_assets/mockup.png)

Live version is available [here](https://pm-detector.herokuapp.com/).

Powdery mildew detection on cherry leaves is a data science and machine learning project with the aim of distinguishing between healthy and diseased cherry leaves. The project includes a binary classification machine learning model that can be used to predict the health status of cherry leaves by uploading images to a Streamlit dashboard. Additionally, the project includes pages with findings from traditional data analysis, a detailed analysis of the hypotheses, and an evaluation of the model's performance.

To ensure a functional pipeline, the project includes three Jupyter notebooks covering data importation and cleaning, data visualization, and the development and evaluation of a TensorFlow deep learning model. These notebooks provide a clear and organized way to manage the project's data and development process.

The project's ultimate goal is to improve the accuracy and efficiency of powdery mildew detection on cherry leaves, which can help growers to identify and treat diseased cherry trees more effectively. By utilizing machine learning and data analysis techniques, this project has the potential to provide an effective and user-friendly solution for identifying healthy and diseased cherry leaves.
___
## Table of Contents

* [Dataset Content](#dataset-content)

* [Business Requirements](#business-requirements)

* [Hypothesis and validation](#hypothesis-and-validation)

* [ML Task Rationale](#ml-task-rationale)

* [ML Business Case](#ml-business-case)

* [Agile Methodology](#agile-methodology)

* [CRISP-DM](#crisp-dm)

* [Dashboard Design](#dashboard-design)

* [ML Model Justification ](#ml-model-justification)

* [Manual Testing](#manual-testing)

* [Features for future consideration](#features-for-future-consideration)

* [Deployment](#deployment)

* [Technologies](#technologies)

* [Credits](#credits)

* [Acknowledgements](#acknowledgements)

___
## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves), backed with a hypothetical scenario from Code Institute in which predictive analytics could be utilised for a practical project case.

* The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.
___
## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
* 3 - The client wants a dashboard that meets the above requirements.
* 4 - The client requires explanations of possible outcomes

___
## Hypothesis and validation

1. Cherry leaves infected with powdery mildew have unique visual characteristics that can be differentiated from healthy cherry leaves using machine learning algorithms. We expect cherry leaves infected with powdery mildew to have a white or greyish powdery coating on the surface, typical symptom is white or greyish marks on the leaves.

**Validation**
The model was able to identify the differences between data points and learn how to accurately predict outcomes based on those distinctions. A model has the ability to predict results on new data without becoming too reliant on the training set. By doing so, the model can generalize its predictions and make reliable forecasts for future observations. Rather than simply memorizing the relationships between features and labels in the training data, the model learns to recognize general patterns, which enhances its predictive power. The model v2 is able to do it with an accuracy of 99%.

2. An ML system trained on cherry leaf images can accurately differentiate between healthy and powdery mildew-infected cherry leaves with at least 90% accuracy. Based on the low complexity of the binary classification task, it is reasonable to expect about 90% accuracy rates.

**Validation**
The hypothesis has been verified by creating a model which has shown excellent performance on the two main evaluation metrics for assessing the business functionality and achievement of the project, namely the overall F1 score of 99% and the recall on the powdery mildew label of 99%.
    
3. Implementing image visualisation in the cherry leaf inspection process will result in a reduction in the misidentification of infected leaves.

**Validation**
Despite the ML model's high accuracy in predicting, the visual features of the disease are hardly distinguishable during the initial stages. Therefore, to reinforce the model and achieve higher final accuracy, a visual examination of cases where the model is not confident is carried out by a staff member during the ML prediction process. This is accomplished using an image and a prediction plot in the Powdery Mildew Detection tab.

 4. The implementation of the ML solution can improve the accuracy and speed of the cherry leaf inspection process, leading to more efficient use of resources and increased productivity and increased worker safety by reducing the amount of time and exposure required for manual inspection of cherry leaves.

**Validation**
 The business case states: 'An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If it has powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. This would mean that it would take 50 hours to inspect and treat 100 trees manually. Now, let's assume that taking pictures of the tree leaves using a smartphone camera and uploading them to the ML model for analysis takes approximately 1 minute per tree. This would mean that it would take 1 hour and 40 minutes to take pictures of all 100 trees and upload them for analysis."

The ML model has 99% accuracy in predicting the health of the trees, the staff member could then focus their attention on the trees that the model has identified as having powdery mildew. This would significantly reduce the time and effort required for manual verification, as the staff member would only need to visually verify a subset of trees rather than all of them. Overall, with the implementation of an ML model, the time required for inspecting and treating 100 trees could be reduced from 50 hours to approximately 2 hours, a savings of 48 hours or a 96 % reduction in time.

5. The use of ML predicting can significantly reduce the company's reliance on manual labour for detecting powdery mildew on cherry leaves, resulting in cost savings and increased efficiency.

**Validation**
We will need to measure the reduction in manual labour needed for detecting powdery mildew on cherry leaves after implementing the ML solution. We can compare the time taken and the cost incurred in the manual inspection process before and after implementing the ML system. Additionally, we can survey the employees involved in the inspection process to gather their feedback on the effectiveness of the ML system in reducing their workload and increasing their efficiency.
____
## ML Task Rationale

*Business Requirement 1:*

As a client, I can navigate easily around an interactive dashboard so that I can view and understand the data presented.
As a client, I can view an image montage of either healthy or powdery mildew-affected cherry leaves so that I can visually differentiate them.
As a client, I can view and toggle visual graphs of average images (and average image difference) and image variabilities for both healthy and powdery mildew-affected cherry leaves so that I can observe the difference and understand the visual markers that indicate leaf quality better.

*Business Requirement 2:*

As a client, I can access and use a machine learning model so that I can obtain a class prediction on a cherry leaf image provided.
As a client, I can provide new raw data on a cherry leaf and clean it so that I can run the provided model on it.
As a client, I can feed cleaned data to the dashboard to allow the model to predict it so that I can instantly discover whether a given cherry leaf is healthy or affected by powdery mildew.
As a client, I can save model predictions in a timestamped CSV file so that I can keep an account of the predictions that have been made.

*Business Requirement 3:*

As a client, I can view an explanation of the project's hypotheses so that I can understand the assumptions behind the machine learning model and its predictions.
As a client, I can view a performance evaluation of the machine learning model so that I can assess its accuracy and effectiveness.
As a client, I can access pages containing the findings from the project's conventional data analysis so that I can gain additional insights into the data and its patterns.

*Business Requirement 4:*

As a client, I can access explanations of possible outcomes from the machine learning model so that I can understand how the model arrived at its predictions and what factors contributed to its decision.

____
## ML Business Case

* The goal is to develop an ML tool that can efficiently and accurately detect if a cherry leaf is healthy or infected with powdery mildew, thus increasing the efficiency of the inspection process and labour quality.

* The dataset provided by the customer will be used to train the ML tool, and the tool's expected output is the ability to accurately differentiate between healthy and infected leaves.

* The customer requires a user-friendly dashboard that allows for the quick uploading of leaf images and at least 97% accuracy in determining if the leaf is healthy or infected.

* To ensure proprietary data remains secure, appropriate measures will be implemented to protect customer data.

* The success of the ML tool will be measured by its accuracy and efficiency in identifying infected leaves, and its ability to reduce the time and cost associated with manual inspection processes.
* The ML tool can be extended to monitor the effectiveness of powdery mildew treatment programs by identifying whether treated leaves return to a healthy state or not.
* The ML tool has the potential to be extended to other crops, such as those that require pest detection, to improve the efficiency and accuracy of the inspection process.
* Validation of the ML tool's performance will be done by training and validating the ML model on a dataset of cherry leaf images with labelled powdery mildew and healthy leaves and evaluating its performance using appropriate metrics such as accuracy or F1 score.
* The model success metric is:
        * A study showing how to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
        * The capability to predict if a cherry leaf is healthy or contains powdery mildew.
        * The model accuracy on test data is over 90%

___
## Agile Methodology

The development has followed an Agile methodology, using GitHub's projects to prioritize and track user stories and features. The approach enabled the implementation of ideas based on their level of importance, ensuring that the website functionality and user experience were not compromised. The following categories were applied, as well as corresponding labels were created:

* must have
* should have
* would have
* could have

![](readme_assets/agile2.png)

The development followed an AGILE approach, which allowed for the delivery of a functional and feature-rich site. The project was constrained by time limitations, which resulted in some initially listed features not being implemented. However, AGILE methodology is incredibly helpful in situations like this, as it allows for the prioritization and tracking of user stories. Completed user stories are in the "Done" section and the ones that were not prioritised for the first iteration are currently in the "To Do" section to be covered in the next iteration.

![](readme_assets/agile1.png)

[See the current state of the project here.](https://github.com/users/oks-erm/projects/3/views/1)

## CRISP-DM

The CRISP-DM approach is a useful framework for developing data mining and AI projects. It helps to align the project goals with the data collection, preparation, modelling, and understanding phases. In the case of the mildew detection project, the CRISP-DM model was repeatedly used to connect the project's hypothesis and model with the stakeholders' requirements.

[!](readme_assets/crisp.png)

**Business Understanding:** The goal of the project was to develop an AI model that could detect powdery mildew on cherry leaves. The project was meant to help farmers detect the disease early and take action to prevent its spread.

**Data Understanding:** The dataset was provided by Code Institute and consisted of images of cherry leaves with and without powdery mildew. The dataset was labelled and had a total of 4000+ images.

**Data Preparation:** The images were preprocessed and cleaned to remove any noise or irrelevant information. The images were also resized and normalized to ensure consistency across the dataset. The dataset was split into training, validation sets and test sets.

**Modeling:** A convolutional neural network (CNN) was chosen as the appropriate model to match the data and produce the desired outcome. The model was developed using TensorFlow and Keras.

**Evaluation:** The model was evaluated using various metrics such as accuracy, precision, recall, and F1 score. The model was also tested on a separate test set to ensure that it was not overfitting the training data.

**Deployment:** The final model was integrated into a Streamlit dashboard that allowed users to upload an image of a cherry leaf and get a prediction of whether it had powdery mildew or not. The dashboard also displayed visualizations of the dataset and the model's performance.

___
## Dashboard Design

1. ***The Summary*** Page of the project provides essential details about the project's background, including its origin and the customer who initiated it. It also lists the business requirements identified by the customer, which define the project's success criteria. Additionally, the page includes the project's objectives and the processes involved in achieving them.

![](readme_assets/dash1.png)
![](readme_assets/dash2.png)

2. ***The Cherry Leaf Visualizer*** page covers the first Data Analysis business objective of the project. It contains plots that can easily be opened and closed via the inbuilt toolbar. 
![](readme_assets/dash3.png)
![](readme_assets/dash4.png)
![](readme_assets/dash5.png)

This app page also includes an image montage creation tool, where the user can select a class of a label for which to display a montage generated via a graphical presentation of random validation set images.
![](readme_assets/dash6.png)
![](readme_assets/dash7.png)

3. ***The Powdery Mildew Detection*** provides a downloadable dataset of infected and uninfected cherry leaf images for live prediction on Kaggle. The user interface includes a file uploader widget that allows the user to upload multiple images of cherry leaves. The system displays the uploaded image, a bar plot of the predicted outcome, and a prediction statement that indicates whether the leaf is infected with powdery mildew and the associated probability. The interface also includes a table that lists the image name and prediction results. Additionally, there is a download button that allows the user to download the report in a .csv format, and a link to the Readme.md file for further information about the project.
![](readme_assets/dash8.png)
![](readme_assets/dash9.png)
![](readme_assets/dash10.png)
![](readme_assets/dash11.png)
![](readme_assets/dash12.png)

4. ***The Hypothesis*** page displays the 5 hypotheses and outcome goals for the project, including success metrics.
![](readme_assets/dash13.png)

5. ***ML Prediction Metrics*** page displays various metrics and visualizations related to the performance of a machine learning model for predicting whether cherry leaves are infected with powdery mildew or not. The first section shows the frequency of labels in the train, validation, and test sets. The second section displays the model training accuracy and losses. The third section shows the general performance of the model on the test set, including the confusion matrix, accuracy, and loss. The fourth section displays the classification report for the model. The final section shows the ROC curve for the model and its AUC score, indicating the model's ability to distinguish between the healthy and powdery mildew classes.
![](readme_assets/dash14.png)
___
## ML Model Justification 

The focus of the development was to build a smaller model that is both robust and accurate for the given task.

### Architecture

The model structure is a convolutional neural network (CNN) that consists of three pairs of convolutional and max pooling layers, followed by a dense layer with a ReLU activation function and a final output layer with a tanh activation function. The CNN architecture is designed to extract spatial features from images by using convolutional filters and pooling layers to reduce the size of the feature maps. The final dense layer aggregates the extracted features and maps them to a binary output using the ReLU activation function, and the final output layer uses the tanh activation function to output a continuous value between -1 and 1, which can be thresholded to binary values.

The number of neurons in the convolutional layers is chosen based on the filters used in each layer. In the first convolutional layer, there are 2 filters and in the second convolutional layer, there are 4 filters. In the third convolutional layer, there are 8 filters. The number of filters in a convolutional layer determines the number of feature maps generated by that layer. Each feature map represents a different feature of the input image. So, having more filters in a layer allows the model to capture more complex features of the input image.

The choice of using 2, 4, and 8 filters in the three convolutional layers is a common approach for image classification tasks, especially when dealing with small datasets. This allows the model to learn a set of filters that can capture low-level features in the initial layers and higher-level features in the later layers.

The hyperparameters of the model are optimized using Keras Tuner, which is a hyperparameter tuning framework that allows the automatic tuning of hyperparameters using various search algorithms. The hyperparameters tuned in this model are the number of units in the dense layer and the learning rate of the optimizer. The optimal number of units in the dense layer is determined by the hp.Int method, which searches for the optimal number of units between 16 and 64 in increments of 16. The optimal learning rate is determined by the hp.Choice method, which selects the learning rate from a set of predefined values.

To prevent overfitting, a dropout layer with a rate of 0.5 is added after the dense layer. A tanh activation function is used for the output layer, which is a good alternative to sigmoid for binary classification tasks. The learning rate used for training the network is also a hyperparameter, which is selected by the hp_learning_rate variable using the hp.Choice method. The Adam optimizer is used to minimize the binary cross-entropy loss function.

Using **tanh** activation and 3 hidden layers for binary classification is a common approach for the following reasons:

* The tanh activation function is a non-linear function that can help capture complex non-linear relationships between the input features and the target variable. A non-linear activation function can improve the model's ability to learn complex patterns in the data, which can be particularly useful in binary classification problems.

* Increasing the number of hidden layers in a neural network can help it learn more complex and abstract representations of the input features. Adding more layers can allow the network to learn multiple levels of abstraction, with each layer building upon the representations learned in the previous layer. This can lead to better performance on complex problems. With the focus on to balancing the complexity of the model with the requirements of the problem and the data, while also avoiding overfitting and improving the generalization of the model. 2-3 hidden layers would be a reasonable depth to start with for binary classification.

* A deeper network can help prevent overfitting by allowing the model to learn multiple levels of abstraction without relying too heavily on any single feature. 

### Optimisation techniques
* Dropout is a regularization technique used to prevent overfitting. It randomly drops out some of the units in the network during training, which helps to prevent the co-adaptation of neurons and encourages the network to learn more robust features.

* Hyperparameter search is the process of finding the optimal values for the hyperparameters of a neural network. In this function, we use two hyperparameters - the number of units in the dense layer and the learning rate of the optimizer. We use the hp.Int method to search for the optimal number of units and the hp.Choice method to select the learning rate from a set of predefined values.

* Regularization is used to prevent overfitting. In this function, we use L2 regularization, which adds a penalty term to the loss function that encourages the weights of the network to stay small.

* Adam Optimizer: Adam is an adaptive optimization algorithm, it uses a combination of momentum and adaptive learning rates to minimize the loss function efficiently.

### Model iterations

### **v1**

![](outputs/v1/model_training_acc.png)
![](outputs/v1/model_training_losses.png)

The model met the requirements of the project but had a relatively low recall score. A relatively high rate of infected leaves misidentified as healthy made me attempt to minimise this metric.

![](readme_assets/clf1.png)
![](outputs/v1/confusion_matrix.png)


### **v2**
The second model accomplished that

![](outputs/v2/confusion_matrix.png)
![](outputs/v2/model_training_acc.png)
![](outputs/v2/model_training_losses.png)

the performance metrics on the training set were superior to the previous iteration, including lower losses.

![](readme_assets/clf.png)
___
## Manual Testing

| Instructions | Expected outcome | Actual Outcome |  |  |
|---|---|---|---|---|
| Navigate the deployed version | The app loads without errors with a Quick Summary Page. | Works as expected |  |  |
| Navigate to the "Project Summary" page  | It provides an overview of the project, including a link to the dataset used, the problem statement, and the approach taken to solve the problem. | Works as expected |  |  |
| Navigate to the "Cherry Leaf Visualiser" page | It provides a section description and three unchecked checkboxes. | Works as expected |  |  |
| Navigate to the "Cherry Leaf Visualiser" page. Check the "Difference between average and variability image" checkbox. | A plot of the mean and variability of images appears along with the observation block. | Works as expected |  |  |
| Navigate to the "Cherry Leaf Visualiser" page. Check the "Difference between average healthy and average powdery mildew cherry leaves" checkbox. | A plot of average healthy, average powdery mildew cherry leaves and the difference appears along with the observation block. | Works as expected |  |  |
| Navigate to the "Cherry Leaf Visualiser" page. Tick the "Image Montage" checkbox. Choose a label from the dropdown menu and click on the button "Create Montage". | The Image Montage section appears, the dropdown menu functions correctly and the montage with the selected label is created. | Works as expected |  |  |
| Navigate to the "Powdery Mildew Detection" page. | It allows the user to upload an image of a cherry leaf. | Works as expected |  |  |
| Click on the "Browse files" button, select an image or a batch from the local machine, and click on the "Upload" button. | The prediction results are displayed on the page along with the confidence score. | Works as expected |  |  |
| Download a prediction report at the bottom of the page. | A prediction report is available to download. | Works as expected |  |  |
| Navigate to the "Project Hypothesis" page. | It provides a high-level summary of the ML project and its expected outcomes. | Works as expected |  |  |
| Navigate to the "ML Prediction Metrics" page. | It provides the evaluation metrics of the machine learning model used in the project. The confusion matrix, precision,  recall, and F1 score, ROC curve, training history of the model are available. | Works as expected |  |  |
| Navigate to the "Usage Guidelines" page. | It provides usage instructions. | Works as expected |  |  |
| Repeat in 3 browsers: Safari, Chrome, and Firefox | The app functions without errors, and the tab icon and background image are displayed. | Works as expected,  except for the tab icon in Safari. |  |  |
|  |  |  |  |  |

___
## Features for future consideration

1. User authentication and storing predicting history for each user.

2. Feautures to allow user feedback.

3. Feature to allow change the result of a prediction manually, if human disagrees with the decision of the model.

4. Automated tests.

____
## Deployment
### Heroku

* The App live link is: https://pm-detector.herokuapp.com/
* The project was deployed on Heroku using the following steps.

1. Log in to Heroku and create an App
2. Log into Heroku CLI in the IDE workspace terminal using the bash command: `heroku login -i` and enter credentials.
3. Run the command `git init` to re-initialise the Git repository
4. Run the command `heroku git:remote -a YOUR_APP_NAME` to connect the workspace and your Heroku app.
5. Set the app's stack to **heroku-20** using the bash command `heroku stack:set heroku-20` to provide compatibility with the Python 3.8.12 version used for this project.
6. Use `git push heroku main` to deploy the application to Heroku.

### Forking the GitHub Project
To create a copy of the GitHub repository to modify and experiment with without affecting the original repository, one can fork the repository:

* On the [repository](https://github.com/oks-erm/ML-mildew-detection) page, navigate to the `Fork` button on the top right corner of the page and click on it to create a copy of the repository which should then appear on your own GitHub account.

### Making a Local Clone

* On the [repository](https://github.com/oks-erm/ML-mildew-detection) page, click on the `Code` button.
* To clone the repository using HTTPS, copy the HTTPS URL.
* Open the IDE of your choice and change the current working directory to the location where you want the cloned directory to be located.
* Type `git clone` and paste the previously copied URL to clone the repository.

___
## Technologies

### Main Data Analysis and Machine Learning Libraries
- [NumPy](https://numpy.org/) - Data processing, preparation and visualisation. Also, TensorFlow is built on top of NumPy
- [Pandas](https://pandas.pydata.org/) - Converting of numerical data into DataFrames
- [Matplotlib](https://matplotlib.org/) - Reading, processing, and displaying image data, producing graphs of tabular data
- [Seaborn](https://seaborn.pydata.org/) - Data visualisation and presentation, such as the confusion matrix heatmap and image dimensions scatter plot.
- [Plotly](https://plotly.com/python/) - Graphical visualisation of data for interactive charts
- [TensorFlow](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf) - Machine learning library used to build the model
- [Keras Tuner](https://keras.io/keras_tuner/) - Tuning of hyperparameters to find the best combination for model accuracy
- [Scikit-learn](https://scikit-learn.org/) - Calculating class weights to handle target imbalance and generating classification report

### Other technologies used
- [Streamlit](https://streamlit.io/) - Library for building interactive web applications for data science - dashboard
- [Heroku](https://www.heroku.com/) - Deployment of the dashboard as a web application
- [Git/GitHub](https://github.com/) - Version control and storing the source code
- [VSCode](https://code.visualstudio.com/) - IDE for local development

____
## Credits 

* Code Institute Malaria Walk Through Project was used for instructional purposes, and guidance throughout the development of this project. For reference and organisation of the app, even though I attempted to optimise the code provided to avoid just copying.

* [Streamlit documentation](https://docs.streamlit.io/) was used for deeper understanding and troubleshooting.

* [Keras Tuner documentation](https://keras.io/keras_tuner/) was used for deeper understanding and troubleshooting.

* [cla-cif/Cherry-Powdery-Mildew-Detector](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector) for reference.

* [Erik1007/mildew-detection-project](https://github.com/Erik1007/mildew-detection-project) for reference.

* [Hyperparameter tuning with Keras Tuner](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html) to learn about hyperparameter tuning.

* [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron

### Media
- The app icon is taken from [Icons8](https://icons8.com/).
- The images used for the design are generated by Stable Diffusion.
____

## Acknowledgements 
* Thank my mentor Marcel Mulders for guidance and support.
* [Claudia Cifaldi](https://github.com/cla-cif) for brainstorming and sharing experience.

____

[Back to the Top](#powdery-mildew-detector-ml-project)