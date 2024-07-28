# Load the dataset from the new file
file_path = 'ai4i2020.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns of the dataset
data.head(), data.columns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'path_to_your_file/ai4i2020.csv'  # Update the file path
data = pd.read_csv(file_path)

# Inspect the dataset columns to identify the target column
print(data.columns)

# Assume 'Machine failure' is the target column based on typical datasets of this nature
target_column = 'Machine failure'

# Preprocess the data
X = data.drop(columns=target_column)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Neural Network': MLPClassifier(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{name} Confusion Matrix')
    plt.show()

# Print Results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Classification Report:\n{result['classification_report']}")
    print(f"ROC AUC: {result['roc_auc']:.2f}\n")
