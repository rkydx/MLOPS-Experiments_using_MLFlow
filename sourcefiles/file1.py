# pip install mlflow
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
max_depth = 5
n_estimators = 10

# Start an MLflow run
with mlflow.start_run():
    # Initialize and train the RandomForestClassifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    

    # Log metrics and model to MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("confusion_matrix.png")

    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    plt.savefig("confusion_matrix.png")
    
    # log artifacts using mlflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)                            # log the current script file as artifact#

    # Tags can be used to add metadata to the run
    # mlflow.set_tag("model_type", "RandomForestClassifier")
    # mlflow.set_tag("dataset", "Wine")
    mlflow.set_tags({
        "Author": "Ramakant", 
        "Project": "MLOps-Experiments-Wine-Classification", 
        "model_type": "RandomForestClassifier", 
        "dataset": "Wine"})
    
    # Log the trained model
    mlflow.sklearn.log_model(clf, "random_forest_model")

    print(f"Model accuracy: {accuracy:.4f}")



