# 🏠 USA House Price Prediction Analysis
## 📌 Project Overview
This project aims to build a robust machine learning pipeline to predict property sale prices. The analysis focuses on handling extreme outliers, mitigating overfitting, and performing business-driven feature engineering to provide accurate valuations for mid-market residential properties.

## ❗ Problem Statement
Accurately pricing real estate is a significant challenge due to several factors:

- High Market Volatility: Property values fluctuate based on complex non-linear relationships between physical attributes and location.

- Impact of Outliers: Extreme property prices (luxury mansions or distressed sales) can heavily bias standard models, leading to poor performance for the average homeowner.

- Skewed Data Distributions: Features like square footage and price often exhibit high skewness, requiring specialized statistical transformations for effective modeling.

This project aims to bridge these gaps by building a generalized model focused on the $0 - $1.5M price segment, ensuring high accuracy where the majority of market transactions occur.

## 📖 Data Dictionary
The dataset contains various features describing the physical attributes, location, and sales timing of properties.

| 🏷️ Feature       | 📖 Description                                                                 |
|------------------|--------------------------------------------------------------------------------|
| 📅 Date          | The date when the property was sold, used for understanding temporal trends.    |
| 💰 Price         | The sale price in USD (Target Variable).                                        |
| 🛏️ Bedrooms      | Number of bedrooms in the property.                                             |
| 🚿 Bathrooms     | Number of bathrooms in the property.                                            |
| 🏠 Sqft Living   | Size of the living area in square feet.                                         |
| 🌳 Sqft Lot      | Total size of the lot in square feet.                                           |
| 🏢 Floors        | Total number of floors in the property.                                         |
| 🌊 Waterfront    | Binary indicator (1: waterfront view, 0: otherwise).                           |
| 👀 View          | Quality index of the property's view (0 to 4).                                  |
| 🛠️ Condition     | Rating of the property's condition (1 to 5).                                    |
| ⬆️ Sqft Above    | Square footage of the property above the basement.                             |
| ⬇️ Sqft Basement | Square footage of the basement.                                                 |
| 🏗️ Yr Built      | The year the property was originally built.                                    |
| 🔧 Yr Renovated  | The year the property was last renovated.                                      |
| 🛣️ Street        | Street address of the property.                                                |
| 🏙️ City          | City where the property is located.                                            |
| 🗺️ Statezip      | State and zip code, providing regional context.                                |
| 🇦🇺 Country       | Country of the property (Note: focuses on properties in Australia).            |


## 🛠️ Data Preprocessing & Feature Engineering
To achieve a high-performing and generalized model, the following steps were implemented:
- Outlier Filtering: Removed properties priced at $0 and focused the scope on houses under $1.5M to improve accuracy for the majority of market participants.
- Log Transformation: Applied $\log(x + 1)$ to the target variable (price) and skewed features (sqft_living, living_per_bedroom) to handle extreme right-skewness and stabilize variance.
- Feature Engineering: Created living_per_bedroom (ratio of living space to bedrooms), which proved to be a high-impact predictor in final models.
- Target Encoding: Transformed high-cardinality geographic features (statezip, city) to capture localized price dynamics effectively.

## 🏆 Model Results
We benchmarked several models to find the optimal balance between accuracy and generalization:
| 🧩 Model             | 📈 R² (Test) | 💵 RMSE (USD) | ⚠️ Gap (Overfit) |
|----------------------|--------------|---------------|------------------|
| 🔹 Ridge Regression  | 0.7173       | $140,025      | 0.0345           |
| 🌟 XGBoost (Selected)| 0.7547       | $126,281      | 0.0791           |
| 🧱 Stacking Regressor| 0.7568       | $126,237      | 0.1681           |

**Decision: XGBoost was chosen as the production model due to its superior RMSE and a healthy variance gap (< 0.10).**


## 📈 Visualizations & Insights
1. **Feature Importance (Gain)**

     Location (statezip) emerged as the most critical driver of property value, followed by size (sqft_living) and the engineered feature living_per_bedroom.

2. **Residual Analysis**

   The residual plot shows errors randomly distributed around zero, and the residual histogram follows a near-perfect normal distribution, confirming that the Log Transformation successfully addressed
   heteroscedasticity.

## 💻 Tech Stack
- Language : Python
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Category Encoders, Matplotlib, Seaborn.
- Environment: Google Colab / Jupyter Notebook.

## 👤 Author
**Baltasar Patrizhard Djata** Informatics Graduate | Aspiring Data Scientist [[linkedin](www.linkedin.com/in/rizhard-djata-a6255131)] | [[Portfolio Link](https://drive.google.com/file/d/1hxXqeZiHD-Lm9TXph4I0WM5_Z90Pxti6/view?usp=drive_link)]

## ⚙️ Installation & Usage
1. Clone this repository:
   `git clone https://github.com/username/usa-house-price-prediction.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the notebook:
   Open `USA_House_Price_Prediction.ipynb` in Google Colab or Jupyter Notebook.
