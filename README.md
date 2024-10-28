# Crop Disease Prediction and Classification Using Environmental and Sensor Data

This project aims to predict and classify crop diseases based on environmental factors and plant condition data. The system analyzes images of crops alongside environmental sensor readings (humidity, temperature, soil moisture, and gas levels) to identify symptoms of plant diseases, enabling early detection and management.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Key Features](#key-features)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Future Improvements](#future-improvements)



## Project Overview

Agricultural diseases can significantly affect crop yield and quality. This project combines environmental data (temperature, humidity, soil moisture, gas levels) with leaf images to detect and classify crop diseases in real time. By identifying patterns between environmental factors and disease symptoms, the system provides an efficient tool for farmers and researchers to diagnose and manage plant health.

## Dataset

The dataset contains approximately 87,000 RGB images of healthy and diseased crop leaves across 38 classes. The data is divided as follows:
- **Training Data**: 80% of images
- **Validation Data**: 20% of images
- **Test Data**: 33 separate images for final testing

### Environmental Features
The system also incorporates environmental sensor data:
- **Humidity**: High levels can promote fungal and bacterial growth.
- **Temperature**: Affects pathogen life cycles and plant susceptibility.
- **Soil Moisture**: Influences root health and the activity of soil pathogens.
- **Gas Levels**: Gases like CO₂ and O₂ indicate plant stress, waterlogging, or anaerobic conditions.

### Symptoms Analysis
The goal is to identify common symptoms associated with diseases by analyzing environmental trends and visual symptoms (e.g., spots, blight, rot) from the leaf images.


## Key Features

- **Automated Image Processing**: Preprocesses leaf images using techniques like resizing, normalization, and feature extraction.
- **Environmental Data Analysis**: Analyzes environmental factors in relation to crop disease susceptibility.
- **Disease Classification Model**: A deep learning model trained on leaf images and sensor data for multi-class disease classification.
- **Multilingual Interface**: The model’s output is available in English and Telugu to increase accessibility for users.




## Model Training and Evaluation

The model is trained on a balanced dataset using ResNet-based architecture for image processing, combined with environmental features as auxiliary inputs. Key metrics:
- **Accuracy**: Evaluates model’s performance across 38 classes.
- **Precision and Recall**: Analyzes model performance for individual diseases.
- **Confusion Matrix**: Visualizes classification results.

## Results

The model achieved **98% accuracy** on the validation set. The top features affecting disease classification are:
1. **Humidity**
2. **Temperature**
3. **Soil Moisture**
4. **Gas Levels**

*Note:* Performance may vary based on environmental conditions and crop type.

## Future Improvements

- **Additional Sensor Data**: Incorporating soil pH and additional gases to improve prediction accuracy.
- **Mobile App Integration**: Developing a mobile-friendly version of the application for on-site disease monitoring.
- **Broader Language Support**: Expanding the app’s multilingual capabilities for better accessibility.





