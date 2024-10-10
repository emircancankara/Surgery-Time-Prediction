# Surgery-Time-Prediction
Here’s an enhanced and more structured **ReadMe** suitable for sharing on GitHub, following standard practices and adding clarity for potential collaborators or viewers:

---

# 🏥 Surgery Time Prediction using Machine Learning

## 📋 Overview

This project is focused on predicting surgery durations based on a variety of patient and operational factors, such as diagnostic codes, anesthesia type, patient demographics, and the medical team involved. By leveraging a machine learning regression model, this project aims to streamline surgery scheduling and resource allocation in medical facilities.

## 📁 Dataset Description

The dataset used in this project includes key features related to surgical procedures and patient information. The main columns are:

- **DiagnosticICD10Code**: International Classification of Diseases, 10th Revision (ICD-10) diagnostic code.
- **SurgeryGroup**: Categorizes surgeries into specific groups for better understanding.
- **AnesthesiaType**: Denotes the type of anesthesia administered.
- **Age**: The age of the patient at the time of the surgery.
- **Sex**: The gender of the patient (0 = Male, 1 = Female).
- **Service**: Represents the type of service provided during the surgery.
- **DoctorID**: A unique identifier for the surgeon performing the procedure.
- **AnaesthetistID**: A unique identifier for the anesthetist involved in the procedure.

The dataset is used to train and test a regression model aimed at predicting the total time of surgery.

## 🧠 Machine Learning Model

A regression model is built to predict surgery time based on the features mentioned above. Key steps involved:
1. **Data Preprocessing**: Handling missing values, encoding categorical data, and scaling.
2. **Model Training**: A regression model is trained to learn patterns from the historical data.
3. **Model Evaluation**: The model’s performance is evaluated using appropriate metrics (e.g., MAE, RMSE).
4. **Prediction**: The trained model is used to predict surgery durations for new data.

### Model Pipeline:
- Data Cleaning and Preprocessing
- Feature Engineering
- Regression Model Training
- Hyperparameter Tuning (if applicable)
- Prediction and Evaluation

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.9+
- Jupyter Notebook
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/surgery-time-prediction.git
   cd surgery-time-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook surgery-time.ipynb
   ```

### Usage

1. Load the dataset into a pandas DataFrame.
2. Preprocess the data (cleaning, encoding, scaling).
3. Train the regression model using the provided code.
4. Evaluate the model’s performance using suitable metrics.
5. Use the model to predict surgery times for new data.

### Example Code Snippet

```python
# Load the dataset
import pandas as pd
data = pd.read_csv('surgery_data.csv')

# Preprocess data (Example: encoding categorical features)
data['AnesthesiaType'] = data['AnesthesiaType'].astype('category').cat.codes

# Train a simple regression model (e.g., Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[['DiagnosticICD10Code', 'SurgeryGroup', 'AnesthesiaType', 'Age', 'Sex']]
y = data['SurgeryTime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Model Evaluation

The model's performance is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

These metrics help in assessing how well the model predicts the surgery duration.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, feel free to:
- Fork the repository
- Create a feature branch
- Submit a Pull Request

Please make sure to update tests as appropriate.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any details, such as adding a specific `requirements.txt` file or additional sections as necessary for your project.
