# Medical-insurance-Predictions

## Problem Statement
Healthcare costs are a significant concern for individuals, families, and insurance providers. Understanding what factors influence medical insurance charges can help insurance companies design fair policies and assist individuals in making informed decisions.

The challenge is to predict the medical insurance cost (charges) based on key factors like age, BMI, smoking habits, and residential region. This project focuses on building a machine learning model to achieve this, while addressing common challenges like data quality, feature engineering, and deployment for user accessibility.

## Objectives
- Analyze the dataset to understand the key factors influencing medical insurance costs.
- Develop a machine learning model capable of predicting charges with high accuracy.
- Build a user-friendly web application to allow individuals or businesses to use the model for predictions.

## Dataset
The dataset contains real-world information about individuals and their corresponding medical insurance costs. It includes:
- `age`: The age of the individual.
- `sex`: Gender of the individual (male/female).
- `bmi`: Body Mass Index (BMI), a measure of body fat based on height and weight.
- `children`: Number of children/dependents covered by the insurance.
- `smoker`: Whether the individual is a smoker (yes/no).
- `region`: Residential region in the United States (southwest, southeast, northwest, northeast).
- `charges`: Final medical insurance cost (target variable).

## Methodology
This project follows a structured methodology for data analysis, modeling, and deployment:

### 1. Data Exploration
- Conducted exploratory data analysis (EDA) to identify trends, correlations, and distributions in the dataset.
- Visualized relationships between features like `smoker` and `charges`, or `bmi` and `charges`, to uncover key insights.

### 2. Data Wrangling
- **Cleaning**: Ensured no missing or inconsistent data in the dataset.
- **Outlier Handling**: Detected and treated outliers in numerical features like `bmi` and `charges` to avoid model bias.
- **Preparation**: Transformed categorical variables like `sex`, `smoker`, and `region` into numerical formats.

### 3. Feature Engineering
- **Feature Scaling**: Applied scaling techniques (e.g., StandardScaler) to normalize numerical features.
- **Feature Selection**: Identified the most important features for prediction using techniques like correlation analysis and feature importance scores.

### 4. Model Development
- Built a **Random Forest Regression** model due to its robustness and ability to handle nonlinear relationships.
- Tuned hyperparameters using grid search for optimal performance.

### 5. Model Evaluation
- Evaluated the model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
- Assessed overfitting/underfitting using cross-validation techniques.

### 6. Deployment
- Created a **Streamlit web application** to allow users to interact with the model.
- The app provides an intuitive interface where users can input details (e.g., age, BMI, smoker status) and receive predicted insurance charges instantly.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Algorithm**: Random Forest Regression
- **Deployment**: Streamlit

