


# Hotel Booking Cancellation Prediction

## Introduction

This project focuses on predicting hotel booking cancellations using machine learning techniques.

**Problem Statement:** The hotel industry faces the challenge of increasing hotel reservation cancellations. This project aims to predict these cancellations, providing valuable insights for both travelers and hotels.





### Data Description

The project utilizes the "Hotel Booking Demand Dataset" from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). This dataset contains 119,390 bookings from two hotels in Portugal spanning from July 2015 to August 2017. It comprises 32 features, including 20 categorical and 12 numerical columns. The target variable for our classification model is "is_cancelled."

### Data Preparation

Before building the model, we performed essential data preparation steps, including:

- Removing duplicate rows.
- Handling missing values by dropping columns and replacing missing values in other columns with the mode.
- Handling outliers.
- Eliminating instances with incorrect values.
- Addressing class imbalance using the SMOTE technique.
- Feature engineering by combining columns with similar meanings to reduce the number of features.

### Data Visualization

We utilized **univariate**, **bivariate**, and **multivariate** data visualization techniques to better understand the data. Generated a correlation matrix to explore relationships within the dataset.

## Methods

### Feature Extraction

We used the **ExtraTreesClassifier()** technique to perform feature extraction. This reduced the number of features from 32 to 14. Extra Trees Classifier is an ensemble learning technique that aggregates results from multiple decision trees to output classifications.

### Handling Imbalanced Data

To address class imbalance, we employed the **Synthetic Minority Oversampling Technique** (SMOTE) to increase instances of minority classes.

### Model Building

We trained two machine learning algorithms: **K-Nearest Neighbors** (KNN) and **Decision Tree**. We also trained these models on data oversampled using SMOTE for comparison.

### Evaluation

The accuracy of our tuned KNN algorithm was 85%, while the tuned Decision Tree algorithm achieved 83%. After applying SMOTE, KNN still outperformed Decision Tree. Classification reports provide further insights into the models' performance.


## Future Work

Potential future work includes:

- Expanding the model to other locations.
- Integrating data from more hotels.
- Incorporating additional features like weather information and social reputation.

Thank you for exploring the Hotel Booking Cancellation Prediction project!
