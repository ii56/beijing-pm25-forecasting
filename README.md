# Urban Air Pollution Forecasting (Beijing PM2.5)

## Overview
This project focuses on forecasting **PM2.5 air pollution levels in Beijing** using historical air quality and meteorological data. Multiple machine learning and deep learning models were developed, evaluated, and compared to identify the most effective approach for short-term air pollution prediction.

The goal of this project is not only high accuracy, but also **clear modeling decisions, interpretability, and real-world relevance**.

---

## Dataset
- **Source:** PRSA Beijing Multi-Site Air Quality Dataset
- **Time range:** March 2013 – February 2017
- **Frequency:** Hourly measurements
- **Stations:** Multiple monitoring sites (e.g., Aotizhongxin)
- **Target variable:** PM2.5 (µg/m³)
- Full citation is below

### Features
- Air pollutants: PM2.5, PM10, SO₂, NO₂, CO, O₃
- Weather variables: Temperature, Pressure, Dew Point, Rain, Wind Speed
- Time-based features: Hour, day, month, day of week
- Lag features and rolling statistics

---

## Project Structure
beijing_pm25_forecasting/
│
├── data/
│   ├── raw/
│   │   └── Dataset/
│   │       └── PRSA2017_Data/
│   │           ├── PRSA_Data_Aotizhongxin.csv
│   │           ├── PRSA_Data_Dongsi.csv
│   │           ├── PRSA_Data_Changping.csv
│   │           └── ... (other monitoring stations)
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_train_test_split.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_tree_based_models.ipynb
│   └── 08_lstm_model.ipynb
│
├── models/
│   ├── random_forest_pm25.pkl
│   ├── gradient_boosting_pm25.pkl
│   └── (optional) other saved models
│
├── usage/
│   |── long_term_projection.ipynb    
│   |── predict_gui.py # Actual usage file to run model
│   └── prep_for_usage.ipynb
│
├── README.md
└── requirements.txt

## How to run the model
Steps:
1. Open the predict_gui.py
2. Run the file
3. Upload the processed csv (The csv must be processed otherwise it would not predict)
4. Click predict to see predictions
5. (Optional) Save csv

---

## Exploratory Data Analysis (EDA)
EDA revealed:
- Strong **seasonal and diurnal patterns** in PM2.5
- High correlation between PM2.5 and PM10, CO, and NO₂
- Inverse relationship between PM2.5 and wind speed
- Small but realistic levels of missing data (~3%)

These insights directly informed **feature engineering and cleaning strategies**.

---

## Data Cleaning Strategy
- Approximately **3.3% of rows** were removed due to missing critical pollutant values
- Weather variables were preserved due to near-complete coverage
- Cleaning choices prioritized **data integrity and temporal consistency**
- Post-cleaning EDA confirmed that overall patterns remained unchanged

---

## Models Implemented
1. Linear Regression (Baseline)
2. Ridge Regression
3. Random Forest
4. Gradient Boosting
5. LSTM (PyTorch)

Tree-based models were included for non-linear learning,
while LSTM was explored for explicit temporal sequence modeling.

---

## Model Evaluation Metrics
Models were evaluated using:

### Mean Absolute Error (MAE)
Average magnitude of prediction error in µg/m³.

### Root Mean Squared Error (RMSE)
Penalizes large errors more strongly; important for extreme pollution events.

### R² (Coefficient of Determination)
Measures how much variance in PM2.5 is explained by the model.

Together, these metrics provide a balanced view of accuracy, robustness, and explanatory power.

---

## Final Model Comparison (Test Set)

| Model               | MAE  | RMSE | R²   |
|---------------------|------|------|------|
| Linear Regression   | 6.00 | 10.22 | 0.988 |
| Ridge Regression    | 6.00 | 10.22 | 0.988 |
| Random Forest       | 3.76 | 7.78  | 0.993 |
| Gradient Boosting   | 5.08 | 8.96  | 0.991 |
| LSTM (PyTorch)      | 16.31| 25.78 | 0.925 |

---

## Key Findings
- **Random Forest** achieved the best overall performance across all metrics
- Gradient Boosting showed strong generalization but slightly lower accuracy
- Linear and Ridge regression performed well as baselines
- LSTM captured trends but underperformed in point prediction accuracy

---

## Final Model Selection
**Random Forest** was selected as the final model due to:
- Lowest prediction error
- Strong generalization
- Robust handling of non-linear relationships
- Lower complexity compared to deep learning approaches

---

## Key Skills Demonstrated
- Time-series data handling
- Feature engineering for environmental data
- Model benchmarking and selection
- Tree-based ML models
- Deep learning experimentation (PyTorch)
- Clear, justifiable modeling decisions

---

## Future Improvements
- Multi-station spatial modeling
- Probabilistic forecasting
- External data integration (traffic, satellite data)
- Advanced sequence models with attention mechanisms

---

## Dataset Citation

This project uses the **Beijing Multi-Site Air Quality Dataset** obtained from the
UCI Machine Learning Repository.

**Citation:**

Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset].  
UCI Machine Learning Repository.  
https://doi.org/10.24432/C5RK5G

**Dataset source:**  
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

---

## Author
This project was developed as a machine learning project with an emphasis on **real-world relevance and professional standards** by Ibaan Ibrahim.
Email: ibaanIbrahim123098@gmail.com

