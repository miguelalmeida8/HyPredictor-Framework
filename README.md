# Failure Prediction and Reporting System

## Overview

Welcome to the repository for the comprehensive Failure Prediction and Reporting System. This project is designed to enhance the reliability and efficiency of industrial operations by leveraging advanced machine learning techniques and domain knowledge. The system aims to mitigate the risks associated with equipment failures, which can lead to significant financial losses and wasted resources.

## Key Features

- **Data Reception and Pre-processing:** Collect and preprocess sensor data to ensure it is clean and ready for analysis.
- **Model Development:** Evaluate and train four machine learning algorithms for accurate failure prediction.
- **Rule-based Adjustments:** Refine initial model predictions using rules derived from domain expertise.
- **Explainability through XAI:** Incorporate Explainable AI techniques to provide transparency and insights into the model's decision-making process.
- **User-implemented Rules:** Allow users to customize the system by adding their own rules based on real-world observations.
- **Failure Reporting and Model Retraining:** Enable users to submit failure reports, triggering model retraining for continuous improvement.

## Performance Highlights

The hybrid approach significantly improves prediction accuracy, achieving an F1 Score of up to 93% and a Recall of 91% in failure prediction. By integrating domain-specific rules, the system's performance was enhanced by up to 28% in the F1 Score for some prediction models. This adaptive system not only boosts prediction accuracy but also fosters proactive maintenance, reducing downtime and operational costs.

## Comparison of Model Performance Before and After Rules

| **Metrics**      | **RF Before** | **RF After** | **XGBoost Before** | **XGBoost After** | **LGBM Before** | **LGBM After** | **CatBoost Before** | **CatBoost After** |
|------------------|---------------|--------------|--------------------|-------------------|-----------------|---------------|---------------------|--------------------|
| Recall (%)       | 90.685        | 90.999       | 91.991             | 91.991            | 88.786          | 90.748        | 91.413              | 91.413             |
| F1 Score (%)     | 64.655        | 93.093       | 90.104             | 90.471            | 92.144          | 92.232        | 89.303              | 90.771             |
| Precision (%)    | 50.236        | 95.285       | 88.293             | 89.001            | 95.767          | 95.855        | 87.288              | 90.137             |
| Accuracy (%)     | 96.586        | 99.535       | 99.305             | 99.333            | 99.479          | 99.547        | 99.247              | 99.547             |

## Demonstration

For a detailed demonstration of how the system works, please watch this video. The video provides an in-depth look at the system's functionality, including real-time failure detection, sensor data visualization, failure report management, rule management, and model explanations.


https://github.com/miguelalmeida8/thesis/assets/115362237/b73bb34d-c52a-4700-bfa6-7c0f4e593218



## How to Use

1. **Clone the Repository:**

2. **Install Dependencies:**

3. **Run the System:**

4. **Access the User Interface:**
Open your web browser and navigate to `http://localhost:5000` to interact with the system.


