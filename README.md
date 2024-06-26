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

The hybrid approach significantly improves prediction accuracy, achieving an F1 Score of up to 90% and a Recall of 92% in failure prediction. By integrating domain-specific rules, the system's performance was enhanced by up to 28% in the F1 Score for some prediction models. This adaptive system not only boosts prediction accuracy but also fosters proactive maintenance, reducing downtime and operational costs.

## Comparison of Model Performance Before and After Rules

| **Metrics**      | **RF Before** | **RF After** | **XGBoost Before** | **XGBoost After** | **LGBM Before** | **LGBM After** | **CatBoost Before** | **CatBoost After** | **GBM Before** | **GBM After** | **Voting Before** | **Voting After** |
|------------------|---------------|--------------|--------------------|-------------------|-----------------|----------------|---------------------|--------------------|----------------|---------------|-------------------|------------------|
| Recall (%)       | 90.685        | 90.999       | 91.991             | 91.991            | 88.786          | 90.748         | 91.413              | 91.413             | 91.003         | 92.242        | 91.003            | 91.413           |
| F1 Score (%)     | 64.655        | 93.093       | 90.104             | 90.471            | 92.144          | 92.232         | 89.303              | 90.771             | 89.576         | 90.245        | 91.055            | 91.278           |
| Precision (%)    | 50.236        | 95.285       | 88.293             | 89.001            | 95.767          | 95.855         | 87.288              | 90.137             | 88.193         | 88.333        | 91.107            | 91.143           |
| Accuracy (%)     | 96.586        | 99.535       | 99.305             | 99.333            | 99.479          | 99.547         | 99.247              | 99.547             | 98.886         | 98.950        | 99.059            | 99.081           |


## Demonstration

For a detailed demonstration of how the system works, please watch this video. The video provides an in-depth look at the system's functionality, including real-time failure detection, sensor data visualization, failure report management, rule management, and model explanations.


https://github.com/miguelalmeida8/HyPredictor-Framework/assets/115362237/a7dfcd6d-442c-4a4c-840c-59564861a939



