# pip install mlflow<3
# pip install seaborn
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for the RandomForestClassifier model
max_depth = 10
n_estimators = 5

# Auto-logging the experiment - metrics, artifacts, model log
mlflow.autolog()

# Mention your experiment below
mlflow.set_experiment('Experiment-1')

# Start an MLflow run
with mlflow.start_run():
    # Initialize and train the RandomForestClassifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
      
    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    plt.savefig("confusion_matrix.png")
    
    # log the current script file as artifact
    mlflow.log_artifact(__file__)                            

    # Tags can be used to add metadata to the run
    mlflow.set_tags({
        "Author": "Ramakant", 
        "Project": "Experiments-Wine-Classification", 
        "model_type": "RandomForestClassifier", 
        "dataset": "Wine"})
    
    print(f"Model accuracy: {accuracy:.4f}")