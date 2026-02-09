# CardioRiskNet: Deep Learning for Cardiovascular Disease Prediction

## Project Overview
**CardioRiskNet** is a deep learning-based system designed to predict the risk of cardiovascular disease using clinical patient data. This project demonstrates the application of neural networks in health science, focusing on data preprocessing, model architecture design, and comprehensive evaluation.

**Disclaimer:** This tool is for educational and research purposes only. It is NOT a medical diagnosis tool and should not be used for clinical decision-making.

---

## 1. Dataset Features & Medical Relevance
We use the **UCI Heart Disease Dataset**, which contains clinical records with the following key features:
- **age**: Patient's age in years.
- **sex**: (1 = male; 0 = female).
- **cp**: Chest pain type (4 values).
- **trestbps**: Resting blood pressure.
- **chol**: Serum cholesterol in mg/dl.
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- **restecg**: Resting electrocardiographic results (values 0, 1, 2).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = yes; 0 = no).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: The slope of the peak exercise ST segment.
- **ca**: Number of major vessels (0-3) colored by flourosopy.
- **thal**: 3 = normal; 6 = fixed defect; 7 = reversable defect.
- **target**: Heart disease (0 = no, 1 = yes).

### Medical Importance
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection based on routine clinical measurements (like cholesterol, blood pressure, and heart rate) can significantly improve patient outcomes through early intervention.

---

## 2. Model Architecture
The system utilizes a Deep Neural Network (DNN) implemented in TensorFlow/Keras:
- **Input Layer**: Receives the normalized clinical features.
- **Hidden Layer 1**: Dense (Fully Connected) with ReLU activation.
- **Hidden Layer 2**: Dense (Fully Connected) with ReLU activation.
- **Dropout Layer**: Reduces overfitting by randomly deactivating neurons during training.
- **Hidden Layer 3**: Dense (Fully Connected).
- **Output Layer**: Sigmoid activation to produce a probability (0 to 1) of heart disease.

---

## 3. Training Process
- **Optimizer**: Adam (Adaptive Moment Estimation).
- **Loss Function**: Binary Cross-Entropy.
- **Metrics**: Accuracy.
- **Validation**: 20% of the data is reserved for validation to monitor for overfitting.

---

## 4. Evaluation Metrics
We evaluate the model using:
- **Accuracy**: Overall correctness.
- **Precision**: Specificity for heart disease detection.
- **Recall (Sensitivity)**: Ability to find all positive cases (critical in medical contexts).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Ability of the model to distinguish between classes.

---

## 5. Ethical Limitations
- **Data Bias**: The dataset may not represent all ethnicities, ages, or genders equally.
- **Interpretability**: Deep learning models are often "black boxes," making it hard to explain *why* a specific risk score was given.
- **Clinical Validation**: Predictions must be validated by licensed medical professionals.
- **Privacy**: Patient data must be handled according to regulations like HIPAA or GDPR.

---

## How to Run
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup and Train (Automated)**:
   ```bash
   python main.py
   ```
   This will clean data, train the model, and generate evaluation reports.

3. **Launch the Web Dashboard**:
   ```bash
   streamlit run app.py
   ```
   This opens an interactive interface for real-time risk prediction.

