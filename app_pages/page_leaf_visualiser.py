import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_leaf_visualiser_body():
    st.write("### Cherry Leaf Visualiser")

    st.info(
        f"* Here you can study a visual difference between healthy and affected"
        f" by powdery mildew cherry leaves, specifically the visual markers for"
        f" defects that define powdery mildew infection."
    )

    st.warning(
        f"We expect cherry leaves infected with powdery mildew have white "
        f"or grayish powdery coating on the surface, typical symptom is a "
        f"white or grayish marks on the leaves. These marks can sometimes "
        f"appear as irregular blotches or spots on the surface of the "
        f"leaf.\n\n"
        f"To apply machine learning techniques to an image dataset, it's "
        f"essential to prepare the images beforehand to achieve optimal "
        f"feature extraction and training. In the case of analyzing powdery "
        f"mildew on a leaf, normalizing the images in the dataset is "
        f"crucial before training a Neural Network on it. Normalizing "
        f"involves calculating the mean and standard deviation of the "
        f"entire dataset, which considers the visual properties of the "
        f"powdery mildew on the leaf. This process enables the machine "
        f"learning model to learn the relevant features accurately and "
        f"efficiently from the image data."
    )

    version = 'v1'
    if st.checkbox("Difference between average and variability image"):

        avg_bad_quality = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_good_quality = plt.imread(
            f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            f"We observed that the average and variability images did not "
            f"display any clear patterns that we could easily distinguish from "
            f"each other. However, leaves that were affected by powdery mildew "
            f"exhibited a greater number of white stripes in the center region."
            )

        st.image(avg_bad_quality,
                 caption='Infected with powdery mildew cherry leaf- Average and Variability')
        st.image(avg_good_quality,
                 caption='Healthy cherry leaf - Average and Variability')
        st.write("---")

    if st.checkbox(
        "Differences between average healthy and average powdery mildew cherry "
        "leaves"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"Differences between average healthy and average powdery mildew "
            f"cherry leaves suggests that healthy cherry leaves form clusters "
            f"of whiteness and brightness, indicating a lighter and brighter "
            f"green color. In contrast, infected leaves form clusters of "
            f"darkness and lower saturation, indicating a less green and less "
            f"saturated appearance.")
        st.image(diff_between_avgs,
                 caption='Difference between average images')
        st.write("---")

    if st.checkbox("Image Montage"):
        st.info(
            f"Using an image montage tool can aid in identifying the specific "
            f"features and variations that the model struggles with, enabling "
            f"the staff member to reinforce the model's accuracy by providing "
            f"targeted feedback. Additionally, the image montage can be used "
            f"to compare features of the correctly predicted cases and the "
            f"incorrectly predicted cases, providing insight into what factors "
            f"may be contributing to the model's errors. This approach allows "
            f"for a more thorough analysis of the model's performance and "
            f"provides valuable information for future model improvements."
        )
        st.write("* To refresh the montage, click on 'Create Montage' button")
        my_data_dir = 'inputs/cherry_leaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select label", options=labels,
                                        index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")

    # Validate inputs
    if not os.path.isdir(dir_path):
        raise ValueError("Invalid directory path.")
    if not isinstance(nrows, int) or not isinstance(ncols, int) \
            or nrows <= 0 or ncols <= 0:
        raise ValueError(
            "Number of rows and columns must be positive integers.")

    labels = os.listdir(dir_path)
    if label_to_display not in labels:
        raise ValueError(
            f"Label {label_to_display} does not exist in directory.")

    images_list = os.listdir(os.path.join(dir_path, label_to_display))
    if nrows * ncols < len(images_list):
        img_idx = random.sample(images_list, nrows * ncols)
    else:
        print(f"To create a montage, reduce the number of rows or columns. "
              f"Your subset contains a total of {len(images_list)} images, "
              f"but you have requested a montage that includes {nrows * ncols}."
              )
        return

    plot_idx = list(itertools.product(range(nrows), range(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for x, img_file in enumerate(img_idx):
        img_path = os.path.join(dir_path, label_to_display, img_file)
        img = imread(img_path)
        img_shape = img.shape
        axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
        axes[plot_idx[x][0], plot_idx[x][1]].set_title(
            f"Width {img_shape[1]}px x Height {img_shape[0]}px")
        axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
        axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

    plt.tight_layout()
    st.pyplot(fig=fig)
