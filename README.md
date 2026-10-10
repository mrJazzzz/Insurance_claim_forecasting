# Car Insurance Claim Frequency Forecasting

## Overview
This project focuses on modeling and predicting the frequency of car insurance claims per policy period using machine learning and actuarial statistical methods. By leveraging policy details, driver demographics, and regional metrics, the goal is to estimate claim frequency while accounting for individual exposure.

## Dataset
The project utilizes an insurance dataset containing information on policyholders and their claim histories over defined exposure periods.

Key attributes included in the dataset:
- `IDpol`: Unique policy identifier.
- `ClaimNb`: Target variable representing the total number of claims registered.
- `Exposure`: Policy duration (in fraction of years), serving as the offset variable.
- `Area`: Area code corresponding to the policyholder's address.
- `VehPower`: Power classification of the vehicle.
- `VehAge`: Age of the vehicle in years.
- `DrivAge`: Age of the primary driver in years.
- `BonusMalus`: Claim history coefficient (bonus-malus rating).
- `VehBrand`: Brand/make category of the vehicle.
- `VehGas`: Fuel type of the vehicle (Diesel or Regular).
- `Density`: Population density of the city where the policyholder resides.
- `Region`: Administrative region of the policyholder.

## Pipeline Architecture

The workflow consists of the following steps:

1. **Data Ingestion and Preprocessing**:
   - Data loading and type conversions.
   - Initial data sanity checks and missing value treatment.
   - Outlier detection and filtering across continuous columns (e.g., capping exposure, driver age, vehicle age, and bonus-malus bounds).

2. **Exploratory Data Analysis (EDA)**:
   - Target and feature distribution visualizations.
   - Binned empirical claim frequency analysis across continuous predictors (Driver Age, Vehicle Age, Density, Bonus-Malus).
   - Frequency breakdown across categorical variables.

3. **Feature Engineering and Diagnostics**:
   - Logarithmic transformations applied to skewed continuous features.
   - Correlation analysis and Variance Inflation Factor (VIF) evaluation to assess multicollinearity.

4. **Modeling**:
   - Generalized Linear Models (GLMs) tailored for count data (e.g., Poisson/Tweedie regression with exposure offsets).
   - Gradient boosted trees (e.g., XGBoost with custom objective/offset handling).
   - Hyperparameter optimization via Bayesian optimization.

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/mrJazzzz/Insurance_claim_forecasting>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Jupyter notebook or main script to execute the data pipeline, EDA, feature diagnostics, and model training:

```bash
jupyter notebook Notebook_task.ipynb
```

## Results and Evaluation
Models are evaluated on test sets using count data metrics (e.g., Poisson Deviance) to accurately measure prediction accuracy relative to policyholder exposure.