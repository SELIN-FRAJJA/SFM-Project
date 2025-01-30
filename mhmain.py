# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
data_path = 'data.csv'  # Update with your dataset path
data = pd.read_csv(data_path)

# Drop rows with missing values
data = data.dropna()

# Encode target labels
label_encoder = LabelEncoder()
data['Risk Level'] = label_encoder.fit_transform(data['Risk Level'])

# Features and target
X = data.drop('Risk Level', axis=1)  # features
y = data['Risk Level']  # target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 150, 186, 200, 250],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Create the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=3,       # Number of cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train the model on the training data using RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best model from the RandomizedSearchCV
best_rf_model = random_search.best_estimator_

# Step 3: Model Evaluation

# Make predictions using the best model
y_pred_rf = best_rf_model.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Print the classification report
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Save the trained model and label encoder
joblib.dump(best_rf_model, 'random_forest_model.pkl')  # Save model
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save label encoder