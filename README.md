# MLOPS-Experiments_using_MLFlow

This repository contains basic ML experiments using **MLflow** for
tracking machine-learning runs.\
Both examples use the **Wine Classification** dataset with a
`RandomForestClassifier`.

## Project Structure

    .
    ├── experiment_localMlflow.py
    ├── experiment_localMLFlow_autolog.py
    └── README.md

## experiment_localMlflow.py (Manual Logging)

This script demonstrates how to manually log all MLflow components:

-   Set MLflow tracking URI (local server)
-   Log parameters (`max_depth`, `n_estimators`)
-   Log metrics (accuracy)
-   Log artifacts (confusion matrix plot, script file)
-   Add tags for metadata
-   Log the trained model manually

This method provides **full control** over what gets tracked.

## experiment_localMLFlow_autolog.py (Auto Logging)

This script uses `mlflow.autolog()` which automatically captures:

-   Parameters\
-   Metrics\
-   Model artifacts\
-   Environment information

You only add extra artifacts (like confusion matrix) if needed.\
Best for **quick experimentation** with minimal code.

## How to Run

### 1. Start MLflow UI

``` bash
mlflow ui
```

Runs at:

    http://127.0.0.1:5000

### 2. Run any experiment

``` bash
python experiment_localMlflow.py
```

or

``` bash
python experiment_localMLFlow_autolog.py
```

### 3. View results in MLflow UI

You can explore:

-   Parameters\
-   Metrics\
-   Confusion matrix artifacts\
-   Tags\
-   Saved models\
-   Run comparisons

## Purpose

This repository helps you learn:

-   How MLflow tracks experiments\
-   Difference between **manual logging** and **auto logging**\
-   How to organize reproducible ML workflows