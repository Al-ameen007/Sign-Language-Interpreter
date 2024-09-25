# Sign-Language-Interpreter

## Project Overview

This project focuses on the development of a system for video-based sign language recognition, utilizing techniques like time-series analysis, sliding windows, and change point detection to accurately classify signs from continuous video data. The approach leverages video frame extraction, landmark detection, and Graph Convolutional Networks (GCN) to achieve efficient and accurate sign recognition.

### Data Preparation

The initial step of the project involves organizing and preparing the dataset. Several utility functions are defined to handle tasks such as:
- Combining folders with data.
- Assigning appropriate labels based on file paths.

This ensures that the training and testing data are structured correctly for the subsequent processing stages.

### Load Data

This section is responsible for loading the training and testing datasets, typically from CSV files. These files contain metadata about the video files, such as their corresponding labels. The data is then prepared as variables for further use in preprocessing and model training.

### Data Preprocessing

The data preprocessing step focuses on handling the raw video files. Key operations include:
- **Frame extraction**: Extracting frames from video sequences.
- **Frame interpolation**: Ensuring uniform frame distribution.
- **Landmark detection**: Detecting facial or hand keypoints for each frame.

This stage prepares the data for time-series analysis, which is crucial for the model to learn temporal dependencies in the sign language videos.

### Model Development

This phase focuses on defining and training a deep learning model for sign language recognition. The steps include:
- **Loading saved models**: Reusing pre-trained models when necessary.
- **Model architecture definition**: Implementing a Graph Convolutional Network (GCN) designed for time-series data.
- **Inference**: Running the model on sample videos and evaluating its performance, particularly its accuracy in recognizing signs.

### Model Evaluation

Once the model is trained, it is evaluated using a test set of videos. Key metrics include:
- **Accuracy**: Calculating the model’s performance on correctly identifying signs.
- **Average time per prediction**: Assessing the computational efficiency during inference.

### Sliding Window and Change Point Detection

The project implements a sliding window technique to handle continuous signing in videos, which allows for the classification of sequential signs. Change point detection algorithms are also integrated to help identify transitions between different signs, improving the model’s ability to accurately segment and classify continuous streams of sign language.
