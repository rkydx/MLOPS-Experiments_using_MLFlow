# pip install mlflow
# pip install seaborn
import os
import mlflow
import mlflow.sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


import dagshub

# -------- Parse Argument for Local Model Saving -------- #
parser = argparse.ArgumentParser()
parser.add_argument("--save-local", action="store_true", help="Keep model folder locally after logging")
args = parser.parse_args()

# Initialize DagsHub + MLflow
dagshub.init(repo_owner='rkydx', repo_name='MLOPS-Experiments_using_MLFlow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/rkydx/MLOPS-Experiments_using_MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model Config - Define the params for the RandomForestClassifier model
max_depth = 10
n_estimators = 5

# Mention your experiment below
mlflow.set_experiment('MLOPS-Exp1')

# Start an MLflow run
with mlflow.start_run():
    # Initialize and train the RandomForestClassifier model
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    

    # Log metrics and model to MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    
    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot using absolute path for safety
    cm_filename = "confusion_matrix.png"
    cm_abspath = os.path.abspath(cm_filename)
    plt.savefig(cm_abspath)
    plt.close()
    
    # log artifacts using mlflow
    #mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(cm_abspath)

    mlflow.log_artifact(os.path.abspath(__file__))                            # log the current script file as artifact#

    # Tags can be used to add metadata to the run
    # mlflow.set_tag("model_type", "RandomForestClassifier")
    # mlflow.set_tag("dataset", "Wine")
    mlflow.set_tags({
        "Author": "Ramakant", 
        "Project": "MLOps-Experiments-Wine-Classification", 
        "model_type": "RandomForestClassifier", 
        "dataset": "Wine"})
    
    # ---------------------------------------------------------------------
    # Instead of calling mlflow.sklearn.log_model (which triggers create_logged_model
    # REST endpoint that DagsHub may not support), save the model locally and then
    # upload the saved model directory as artifacts. This avoids unsupported endpoints.
    # ---------------------------------------------------------------------
    model_dir = "random_forest_model_local"

    # make sure local model_dir is clean
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # save the model locally using mlflow.sklearn.save_model
    mlflow.sklearn.save_model(clf, path=model_dir)

    # Log the trained model
    # mlflow.sklearn.log_model(clf, "random_forest_model")

    # upload the saved model directory as artifacts to the run (artifact_path will be shown in UI)
    try:
        mlflow.log_artifacts(model_dir, artifact_path="random_forest_model")
    except Exception as e:
        # If remote server rejects artifact upload, print the error for debugging
        print("Failed to log artifacts:", e)

    # Remove Local Model Folder Unless --save-local Used like as below:
    # python .\sourcefiles\experiment_remoteDAGSHub.py --save-local
    if args.save_local:
        print(f"Local model saved at: {os.path.abspath(model_dir)}")
    else:
        shutil.rmtree(model_dir)
        print("Local model folder deleted (use --save-local to keep it)")

    print(f"Model accuracy: {accuracy:.4f}")
    print("Run complete. View run in DagsHub/MLflow UI.")


