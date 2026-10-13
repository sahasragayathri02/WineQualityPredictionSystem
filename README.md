# 🍷 Wine Quality Prediction System

### Machine Learning Project | Binary & Multinomial Logistic Regression

A Python-based Machine Learning application that analyzes the physicochemical properties of wine and predicts its quality using **Binary Logistic Regression** and **Multinomial Logistic Regression**.

The project combines data preprocessing, model training, performance evaluation, feature analysis, and an interactive command-line interface into one complete workflow.


---

## 📖 Project Overview

Wine quality assessment is traditionally performed by experts through sensory evaluation. However, physicochemical measurements can also provide useful information for analyzing wine quality.

The **Wine Quality Prediction System** applies supervised Machine Learning techniques to predict wine quality based on measurable chemical properties.

The system uses a wine dataset containing 11 physicochemical input features and a quality score as the target variable.

Two Logistic Regression approaches are implemented:

1. **Binary Logistic Regression** — classifies wine into two quality categories.
2. **Multinomial Logistic Regression** — predicts among multiple quality categories.

The application provides an interactive command-line interface through which users can explore the dataset, inspect model performance, view feature importance, test sample records, and generate predictions for CSV files.

The project demonstrates how Machine Learning concepts can be applied to a practical classification problem.

## 🎯 Objectives

The main objectives of this project are:

- To understand and implement supervised Machine Learning.
- To preprocess and analyze a real-world wine quality dataset.
- To implement Binary and Multinomial Logistic Regression.
- To predict wine quality using physicochemical characteristics.
- To compare classification models using standard evaluation metrics.
- To analyze the contribution of input features using model coefficients.
- To support sample-based and batch predictions.
- To build a modular, menu-driven Python application.
- To save trained models and evaluation outputs for reuse.

## 📊 Dataset Description

The dataset contains physicochemical measurements of wine samples and their corresponding quality scores.

### Input features

| No. | Feature | Description |
|---:|---|---|
| 1 | Fixed Acidity | Acids that do not evaporate easily |
| 2 | Volatile Acidity | Acids associated with volatile compounds |
| 3 | Citric Acid | Citric acid content in the wine |
| 4 | Residual Sugar | Sugar remaining after fermentation |
| 5 | Chlorides | Chloride concentration |
| 6 | Free Sulfur Dioxide | Free sulfur dioxide concentration |
| 7 | Total Sulfur Dioxide | Total sulfur dioxide concentration |
| 8 | Density | Density of the wine |
| 9 | pH | Measure of acidity or alkalinity |
| 10 | Sulphates | Sulphate concentration |
| 11 | Alcohol | Alcohol content |

**Target variable:** `quality`

The original quality score is used to derive the classification target required by each model.

The dataset is intended for educational and analytical use. Predictions represent patterns learned from the available data and are not a substitute for professional wine tasting or laboratory analysis.

## 🤖 Machine Learning Algorithms

### 1. Binary Logistic Regression

Binary Logistic Regression is a supervised classification algorithm used when the target has two possible classes.

It estimates the probability that a sample belongs to a particular class using the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The predicted probability is converted into a class using a decision threshold.

In this project, wine quality is represented using two categories derived from the original quality scores. The configured preprocessing and classification rules determine the exact class boundary.

### 2. Multinomial Logistic Regression

Multinomial Logistic Regression extends Logistic Regression to problems containing more than two classes.

It uses the softmax function to estimate probabilities across multiple categories:

\[
P(y=k\mid x)=
\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

The class with the highest predicted probability is selected as the model's prediction.

In this project, the model predicts among multiple wine quality categories represented in the processed dataset.

### Model Comparison

Both models are evaluated on held-out test data to understand their classification performance.

The comparison considers:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix, where available
- Model coefficients and feature contributions

The actual performance depends on the dataset, preprocessing configuration, classification boundaries, and training results.

## ✨ Key Features

### Interactive Command-Line Interface

The application provides a menu-driven interface for accessing its prediction, analysis, and evaluation functions.

### Sample Predictions

- Demonstrate predictions using held-out test samples.
- Select available sample records.
- Generate predictions using randomly selected records.
- View actual and predicted quality information where available.

### Batch CSV Prediction

- Load a CSV file containing wine measurements.
- Process multiple records in one operation.
- Generate predictions for the input records.
- Export the results to a CSV output file.

The batch workflow uses the expected feature columns rather than requiring users to manually enter each physicochemical measurement.

### Model Evaluation

- View the performance of Binary Logistic Regression.
- View the performance of Multinomial Logistic Regression.
- Compare available evaluation metrics.
- Inspect saved model evaluation results.

### Feature Analysis

- Examine model coefficients.
- Explore the relative influence of physicochemical features.
- Understand how learned coefficients relate to predicted classes.

For Logistic Regression, coefficients describe learned relationships on the model's feature scale; they should not automatically be interpreted as causal effects.

### Dataset Exploration

- View dataset statistics.
- Inspect quality distributions and available records.
- Search for samples belonging to particular quality categories.
- Explore relevant dataset information through the menu.

## 🔄 Project Workflow

The project follows this general Machine Learning pipeline:

1. **Data loading:** Load the wine dataset into Python.
2. **Data inspection:** Examine columns, data types, missing values, duplicates, and target distribution.
3. **Preprocessing:** Prepare the feature matrix and classification targets.
4. **Train-test split:** Separate data for model training and held-out evaluation.
5. **Model training:** Train Binary and Multinomial Logistic Regression models.
6. **Model evaluation:** Calculate relevant classification metrics.
7. **Model persistence:** Save trained models for later use.
8. **Prediction:** Apply the trained models to supported sample records or CSV inputs.
9. **Result generation:** Display predictions and save applicable outputs.

The project uses an 80:20 train-test split, stratification where applicable, and a fixed random state of `42` for reproducibility.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core programming language |
| pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical computation |
| scikit-learn | Model training, preprocessing, and evaluation |
| joblib | Saving and loading trained models |
| Matplotlib / Seaborn | Visualization, if included in the implementation |
| CSV and JSON | Storing prediction results and evaluation information |
| Command-Line Interface | User interaction |

## 📁 Project Structure

The project is organized into modules to separate configuration, data preparation, model training, and prediction logic.

```text
wine-quality-ml/
│
├── data/
│   └── winequality.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── predict.py
│
├── models/
│   └── Saved model artifacts (.joblib)
│
├── outputs/
│   └── predictions/
│       └── batch_predictions.csv
│
├── run_project.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** This is the logical project layout. Actual filenames, model artifact locations, and optional modules should match the files present in your repository.

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or a compatible version supported by your dependencies.
- pip, the Python package installer.
- Git, if you want to clone the repository.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd wine-quality-ml
```

Replace `<YOUR_REPOSITORY_URL>` with your actual GitHub repository URL.

If you have already downloaded the project, open a terminal in the project directory instead.

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows — Command Prompt**

```bat
.venv\Scripts\activate.bat
```

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is missing, install the dependencies actually used by the project, such as:

```bash
pip install pandas numpy scikit-learn joblib
```

### Step 5: Verify the Dataset and Saved Models

Ensure the dataset is available at the location configured in `src/config.py`.

If the project expects pre-trained `.joblib` files, verify that the required model artifacts are present. Alternatively, train the models using the project training script.

## ▶️ How to Run

### Option 1: Launch the Prediction Interface

From the project root directory, run:

```bash
python3 -m src.predict
```

This launches the interactive command-line interface, provided the configured dataset and required model artifacts are available.

### Option 2: Train or Rebuild the Project

```bash
python3 run_project.py
```

Use this command to execute the training workflow implemented by the project script.

If the script trains the models from scratch, it should prepare the data, fit the models, evaluate their performance, and save the resulting artifacts according to the configured paths.

**Important:** Training and prediction are separate workflows. The prediction interface may require trained model files to exist before it can run successfully.

## 🧭 Available Modules

The command-line application is designed around the following functionality.

| Module | Description |
|---|---|
| 1. Sample Prediction Demo | Demonstrates prediction on a held-out test sample |
| 2. Select a Sample | Allows prediction on an available sample record |
| 3. Random Sample Prediction | Selects a random record for prediction |
| 4. Batch CSV Prediction | Processes multiple records from a CSV file |
| 5. Model Comparison | Compares available model performance |
| 6. Performance Metrics | Displays classification evaluation metrics |
| 7. Feature Importance / Coefficients | Inspects learned Logistic Regression coefficients |
| 8. Dataset Statistics | Displays descriptive statistics and dataset information |
| 9. Quality-Based Search | Searches records using quality-related criteria |
| 10. Exit | Closes the application |

The exact menu labels and numbering depend on the implementation in `src/predict.py`.

## 📈 Model Evaluation

The models are evaluated using held-out test data.

### Accuracy

Measures the proportion of all predictions that are correct.

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

Measures how many positive predictions are correct.

\[
Precision=\frac{TP}{TP+FP}
\]

### Recall

Measures how many actual positive samples are correctly identified.

\[
Recall=\frac{TP}{TP+FN}
\]

### F1-Score

Combines precision and recall into a single metric.

\[
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
\]

For multiclass classification, precision, recall, and F1-score may be reported using macro, micro, or weighted averaging. The averaging method should be stated alongside the results.

### Performance Comparison

The application can display the metrics recorded during model evaluation. Actual scores should be generated from the current saved evaluation results rather than estimated or manually assumed.

| Metric | Binary Logistic Regression | Multinomial Logistic Regression |
|---|---|---|
| Accuracy | Generated by evaluation | Generated by evaluation |
| Precision | Generated by evaluation | Generated by evaluation |
| Recall | Generated by evaluation | Generated by evaluation |
| F1-score | Generated by evaluation | Generated by evaluation |

## 📂 Output Files

The project can generate or maintain the following artifacts, depending on the configured workflow.

- **Saved models:** Trained `.joblib` files used for prediction.
- **Batch prediction results:** `outputs/predictions/batch_predictions.csv`
- **Evaluation reports:** Saved metrics in JSON or CSV format, where configured.
- **Coefficient reports:** Saved model coefficient information, where configured.

Generated files should be stored in their configured output directories. Ensure those directories exist or are created by the application.

## ⚠️ Limitations

- Prediction quality depends on the dataset and its representation of real-world wine samples.
- Classification results depend on how the original quality scores are converted into categories.
- Logistic Regression may not capture complex nonlinear relationships between physicochemical features.
- Performance on unseen data may differ from held-out test performance.
- Model coefficients indicate learned statistical relationships and do not establish causation.
- The system is intended for educational and experimental purposes rather than professional wine certification.

## 🚀 Future Enhancements

Potential improvements include:

- Developing a web interface using Flask or Streamlit.
- Adding Decision Tree, Random Forest, and other classification models.
- Improving class imbalance handling.
- Performing cross-validation and hyperparameter tuning.
- Adding visualizations for feature distributions and model performance.
- Providing a downloadable prediction report.
- Using explainability tools to improve prediction transparency.
- Packaging the application for easier installation and deployment.

## 👩‍💻 Authors

**Machine Learning Project — Wine Quality Prediction System**

Developed as an academic project to explore supervised classification, Logistic Regression, data preprocessing, model evaluation, and practical Machine Learning implementation.

**Contributors:** 
Ch Sahasra Gayathri 
Antony Nissy 

## 📜 License

This project is intended for educational and academic use. Add a license file, such as the MIT License, only if you intend to distribute the project under those terms.

---

⭐ **If you find this project useful, consider starring the repository!**
