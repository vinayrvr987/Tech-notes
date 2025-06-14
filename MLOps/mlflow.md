# MLflow Complete Guide

## Table of Contents
1. [What is MLflow?](#what-is-mlflow)
2. [Core Components](#core-components)
3. [MLflow Tracking](#mlflow-tracking)
4. [MLflow Projects](#mlflow-projects)
5. [MLflow Models](#mlflow-models)
6. [MLflow Model Registry](#mlflow-model-registry)
7. [Deployment Strategies](#deployment-strategies)
8. [Best Practices](#best-practices)
9. [Integration with Popular Tools](#integration-with-popular-tools)
10. [Common Interview Questions](#common-interview-questions)
11. [Hands-on Examples](#hands-on-examples)
12. [Advanced Topics](#advanced-topics)

---

## What is MLflow?

**MLflow** is an open-source platform for managing the complete machine learning lifecycle. It addresses four primary functions:

- **Experiment Tracking**: Log and compare experiments
- **Code Packaging**: Package ML code for reproducible runs
- **Model Management**: Deploy models to various platforms  
- **Model Registry**: Centralized model store with versioning

### Key Benefits
- **Reproducibility**: Track experiments with full lineage
- **Collaboration**: Share experiments and models across teams
- **Deployment**: Deploy models to various platforms easily
- **Version Control**: Track model versions and transitions
- **Framework Agnostic**: Works with any ML library (scikit-learn, TensorFlow, PyTorch, etc.)

---

## Core Components

### 1. MLflow Tracking
- Records and queries experiments
- Logs parameters, metrics, and artifacts
- Provides UI for visualization

### 2. MLflow Projects
- Packages ML code in reusable format
- Defines dependencies and entry points
- Enables reproducible runs across environments

### 3. MLflow Models
- Standard format for packaging ML models
- Supports multiple deployment targets
- Includes model signature and dependencies

### 4. MLflow Model Registry
- Centralized model store
- Model versioning and stage transitions
- Collaborative model management

---

## MLflow Tracking

### Core Concepts

**Run**: Single execution of ML code
**Experiment**: Collection of related runs
**Artifact**: Files (models, plots, data files)
**Parameter**: Input to your model (hyperparameters)
**Metric**: Evaluation measure (accuracy, loss, etc.)

### Basic Usage

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log artifacts (plots, data files)
    mlflow.log_artifact("confusion_matrix.png")
```

### Tracking Server Setup

```bash
# Local tracking server
mlflow server --host 0.0.0.0 --port 5000

# With backend store and artifact store
mlflow server \
    --backend-store-uri postgresql://user:password@localhost:5432/mlflow \
    --default-artifact-root s3://my-mlflow-bucket/ \
    --host 0.0.0.0 \
    --port 5000
```

### Setting Tracking URI

```python
import mlflow

# Set tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Or use environment variable
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
```

### Advanced Tracking Features

```python
# Nested runs for hyperparameter tuning
with mlflow.start_run(run_name="hyperparameter_tuning"):
    for alpha in [0.1, 0.5, 1.0]:
        with mlflow.start_run(nested=True):
            mlflow.log_param("alpha", alpha)
            # Train and log model
            
# Automatic logging (framework-specific)
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog()
mlflow.pytorch.autolog()

# Custom tags
mlflow.set_tag("team", "data-science")
mlflow.set_tag("version", "v1.2.3")

# Logging artifacts from different sources
mlflow.log_artifacts("path/to/directory", artifact_path="data")
mlflow.log_dict({"config": "value"}, "config.json")
mlflow.log_text("Some text content", "notes.txt")
```

---

## MLflow Projects

### Project Structure
```
my_ml_project/
├── MLproject
├── conda.yaml
├── requirements.txt
├── train.py
└── predict.py
```

### MLproject File
```yaml
name: wine-quality

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.5}
    command: "python train.py --alpha {alpha} --l1_ratio {l1_ratio}"
  
  predict:
    parameters:
      model_uri: {type: string}
      input_data: {type: string}
    command: "python predict.py --model-uri {model_uri} --input-data {input_data}"
```

### Environment Files

**conda.yaml**
```yaml
name: wine-quality
channels:
  - conda-forge
dependencies:
  - python=3.8
  - scikit-learn=1.0.2
  - pandas=1.4.2
  - numpy=1.21.5
  - pip
  - pip:
    - mlflow>=1.26.0
```

**requirements.txt**
```
scikit-learn==1.0.2
pandas==1.4.2
numpy==1.21.5
mlflow>=1.26.0
```

### Running Projects

```bash
# Run from local directory
mlflow run . -P alpha=0.8 -P l1_ratio=0.3

# Run from Git repository
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.5

# Run specific entry point
mlflow run . -e predict -P model_uri=runs:/abc123/model -P input_data=data.csv

# Run with specific environment
mlflow run . --env-manager=conda
mlflow run . --env-manager=virtualenv
```

---

## MLflow Models

### Model Format

MLflow models are stored as directories with:
- **MLmodel**: Metadata file describing the model
- **Model files**: Serialized model artifacts
- **Requirements**: Dependencies and environment info

### MLmodel File Example
```yaml
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.8.10
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.0.2
model_uuid: 1234567890abcdef
signature:
  inputs: '[{"name": "feature1", "type": "double"}, {"name": "feature2", "type": "double"}]'
  outputs: '[{"name": "prediction", "type": "double"}]'
```

### Model Signatures

```python
from mlflow.models.signature import infer_signature
import pandas as pd

# Infer signature from training data
signature = infer_signature(X_train, model.predict(X_train))

# Manual signature creation
from mlflow.types import Schema, ColSpec
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
])
output_schema = Schema([ColSpec("double", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log model with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### Model Loading and Serving

```python
# Load model as Python function
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(X_test)

# Load native format
sklearn_model = mlflow.sklearn.load_model(model_uri)

# Model URIs
# runs:/<run_id>/<artifact_path>
# models:/<model_name>/<version>
# models:/<model_name>/<stage>
# file:///path/to/model
# s3://bucket/path/to/model
```

### Built-in Model Serving

```bash
# Serve model locally
mlflow models serve -m runs:/abc123/model -p 1234

# Serve with specific environment
mlflow models serve -m models:/wine-quality/1 --env-manager conda

# Test the served model
curl -X POST -H "Content-Type:application/json" \
  --data '{"instances": [[1, 2, 3, 4]]}' \
  http://localhost:1234/invocations
```

---

## MLflow Model Registry

### Model Registry Workflow

1. **Register Model**: Add model to registry
2. **Version Management**: Track model iterations  
3. **Stage Transitions**: Move models through lifecycle
4. **Annotations**: Add descriptions and tags

### Registering Models

```python
# Register model during logging
mlflow.sklearn.log_model(
    model, 
    "model", 
    registered_model_name="wine-quality-model"
)

# Register existing model
model_uri = "runs:/abc123/model"
mlflow.register_model(model_uri, "wine-quality-model")
```

### Model Stages

- **None**: Initial stage
- **Staging**: Testing stage
- **Production**: Live deployment
- **Archived**: Deprecated models

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to staging
client.transition_model_version_stage(
    name="wine-quality-model",
    version=1,
    stage="Staging"
)

# Add description
client.update_model_version(
    name="wine-quality-model",
    version=1,
    description="Random Forest model with improved accuracy"
)

# Get latest model by stage
latest_version = client.get_latest_versions(
    "wine-quality-model", 
    stages=["Production"]
)[0]
```

### Model Registry UI Operations

```python
# List registered models
client.list_registered_models()

# Get model details
model = client.get_registered_model("wine-quality-model")

# Get specific version
version = client.get_model_version("wine-quality-model", "1")

# Delete model version
client.delete_model_version("wine-quality-model", "1")

# Delete entire model
client.delete_registered_model("wine-quality-model")
```

---

## Deployment Strategies

### 1. Real-time Serving

```bash
# MLflow built-in serving
mlflow models serve -m models:/wine-quality/Production -p 5000

# Docker deployment
mlflow models build-docker -m models:/wine-quality/1 -n wine-model
docker run -p 5000:8080 wine-model

# Cloud deployment
mlflow deployments create -t sagemaker -m models:/wine-quality/1 --name wine-deployment
```

### 2. Batch Inference

```python
import mlflow.pyfunc

# Load model
model = mlflow.pyfunc.load_model("models:/wine-quality/Production")

# Batch prediction
import pandas as pd
batch_data = pd.read_csv("batch_input.csv")
predictions = model.predict(batch_data)

# Save results
results = pd.DataFrame({
    'predictions': predictions,
    'input_data': batch_data.to_dict('records')
})
results.to_csv("batch_predictions.csv")
```

### 3. Streaming Deployment

```python
# Apache Kafka integration
from kafka import KafkaConsumer, KafkaProducer
import json

model = mlflow.pyfunc.load_model("models:/wine-quality/Production")

consumer = KafkaConsumer('input-topic')
producer = KafkaProducer('output-topic')

for message in consumer:
    data = json.loads(message.value)
    prediction = model.predict([data['features']])
    
    result = {
        'id': data['id'],
        'prediction': prediction[0],
        'timestamp': time.time()
    }
    
    producer.send('output-topic', json.dumps(result))
```

---

## Best Practices

### 1. Experiment Organization

```python
# Use meaningful experiment names
mlflow.set_experiment("wine-quality-random-forest")

# Consistent naming conventions
with mlflow.start_run(run_name="rf_100_estimators_v1"):
    pass

# Use tags for organization
mlflow.set_tag("model_type", "random_forest")
mlflow.set_tag("feature_set", "v2")
mlflow.set_tag("data_version", "2023-01-15")
```

### 2. Parameter and Metric Logging

```python
# Log all relevant parameters
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "feature_selection": "recursive",
    "data_preprocessing": "standard_scaler"
}
mlflow.log_params(params)

# Log comprehensive metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc_roc": auc_roc,
    "training_time": training_time
}
mlflow.log_metrics(metrics)

# Log metrics over time (for training curves)
for epoch in range(num_epochs):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

### 3. Artifact Management

```python
# Save and log important artifacts
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True)
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance.to_csv("feature_importance.csv", index=False)
mlflow.log_artifact("feature_importance.csv")

# Model artifacts with organization
mlflow.log_artifacts("preprocessing/", artifact_path="preprocessing")
mlflow.log_artifacts("model_analysis/", artifact_path="analysis")
```

### 4. Environment Management

```python
# Log environment information
import sys
import sklearn
import pandas as pd

mlflow.log_param("python_version", sys.version)
mlflow.log_param("sklearn_version", sklearn.__version__)
mlflow.log_param("pandas_version", pd.__version__)

# Save full environment
mlflow.log_artifact("requirements.txt")
```

### 5. Model Documentation

```python
# Model with comprehensive metadata
model_info = {
    "description": "Random Forest classifier for wine quality prediction",
    "training_data": "wine-quality-dataset-v2.csv",
    "features": list(X_train.columns),
    "target": "quality",
    "model_type": "RandomForestClassifier",
    "hyperparameters": params,
    "performance_metrics": metrics,
    "training_date": datetime.now().isoformat(),
    "data_preprocessing": ["standard_scaling", "feature_selection"],
    "known_limitations": ["Sensitive to class imbalance", "Limited to tabular data"]
}

mlflow.log_dict(model_info, "model_documentation.json")
```

---

## Integration with Popular Tools

### 1. Docker Integration

```dockerfile
# Dockerfile for MLflow model
FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ model/
EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "model/", "-h", "0.0.0.0", "-p", "8080"]
```

### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlflow-model
  template:
    metadata:
      labels:
        app: mlflow-model
    spec:
      containers:
      - name: model
        image: wine-quality-model:latest
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
```

### 3. Apache Airflow Integration

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow

def train_model(**context):
    with mlflow.start_run():
        # Training logic
        pass

def deploy_model(**context):
    # Get best model from experiment
    experiment = mlflow.get_experiment_by_name("wine-quality")
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])
    best_run = runs.iloc[0]
    
    # Deploy best model
    model_uri = f"runs:/{best_run.run_id}/model"
    mlflow.register_model(model_uri, "wine-quality-production")

dag = DAG('ml_pipeline', schedule_interval='@daily')

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

train_task >> deploy_task
```

### 4. GitHub Actions CI/CD

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python train.py
    
    - name: Deploy model
      if: github.ref == 'refs/heads/main'
      run: |
        python deploy.py
```

---

## Common Interview Questions

### Technical Questions

**Q1: How does MLflow differ from other ML platforms like Kubeflow or SageMaker?**

*Answer*: MLflow is framework-agnostic and focuses on the ML lifecycle management, while Kubeflow is Kubernetes-native for ML workflows and SageMaker is AWS-specific. MLflow provides better flexibility across cloud providers and local environments.

**Q2: Explain the MLflow Model format and its benefits.**

*Answer*: MLflow Models use a standardized directory structure with an MLmodel file containing metadata, model artifacts, and environment specifications. Benefits include reproducibility, multi-framework support, and deployment flexibility.

**Q3: How would you implement A/B testing with MLflow?**

```python
# A/B Testing implementation
def ab_test_deployment():
    # Deploy model A (control)
    model_a = mlflow.pyfunc.load_model("models:/wine-quality/Production")
    
    # Deploy model B (treatment)  
    model_b = mlflow.pyfunc.load_model("models:/wine-quality/Staging")
    
    # Route traffic based on user ID
    def route_prediction(user_id, features):
        if hash(user_id) % 100 < 50:  # 50% traffic to each
            prediction = model_a.predict([features])[0]
            model_version = "A"
        else:
            prediction = model_b.predict([features])[0]
            model_version = "B"
            
        # Log prediction for analysis
        with mlflow.start_run():
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("user_id", user_id)
            mlflow.log_metric("prediction", prediction)
            
        return prediction, model_version
```

**Q4: How do you handle model versioning and rollbacks?**

*Answer*: Use MLflow Model Registry stages (Staging, Production, Archived) with systematic version transitions. Implement automated rollback based on performance metrics monitoring.

**Q5: Describe your approach to monitoring model drift with MLflow.**

```python
# Model drift monitoring
def monitor_data_drift():
    # Load reference data (training data)
    reference_data = mlflow.artifacts.download_artifacts("runs:/ref_run_id/training_data.csv")
    
    # Get recent predictions and inputs
    recent_data = get_recent_inference_data()
    
    # Calculate drift metrics
    drift_score = calculate_drift(reference_data, recent_data)
    
    # Log drift metrics
    with mlflow.start_run():
        mlflow.log_metric("data_drift_score", drift_score)
        mlflow.log_metric("prediction_drift", prediction_drift)
        
        if drift_score > threshold:
            mlflow.set_tag("drift_alert", "high")
            trigger_retraining()
```

### Scenario-Based Questions

**Q6: How would you set up MLflow for a team of 20 data scientists?**

*Answer*: 
- Deploy MLflow tracking server with PostgreSQL backend
- Use S3/Azure Blob for artifact storage
- Implement RBAC with authentication
- Set up automated backup and monitoring
- Establish naming conventions and best practices
- Create shared experiment templates

**Q7: Describe how you'd implement automated model retraining.**

```python
# Automated retraining pipeline
def automated_retraining():
    # Check performance metrics
    current_model_performance = get_production_metrics()
    
    if current_model_performance < performance_threshold:
        # Trigger retraining
        with mlflow.start_run():
            # Get latest data
            new_data = get_latest_training_data()
            
            # Retrain model
            new_model = train_model(new_data)
            
            # Evaluate new model
            performance = evaluate_model(new_model)
            
            # Log everything
            mlflow.log_metrics(performance)
            mlflow.sklearn.log_model(new_model, "model")
            
            # Register if better
            if performance > current_model_performance:
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", 
                                    "wine-quality-model")
```

---

## Hands-on Examples

### Complete End-to-End Example

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLflowExperiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def prepare_data(self, data_path):
        """Load and prepare data"""
        self.data = pd.read_csv(data_path, sep=';')
        
        # Feature engineering
        self.data['alcohol_sugar_ratio'] = self.data['alcohol'] / (self.data['residual sugar'] + 1)
        self.data['acid_ratio'] = self.data['fixed acidity'] / self.data['volatile acidity']
        
        # Prepare features and target
        X = self.data.drop('quality', axis=1)
        y = (self.data['quality'] >= 6).astype(int)  # Binary classification
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning with MLflow tracking"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        with mlflow.start_run(run_name="hyperparameter_tuning"):
            mlflow.set_tag("stage", "hyperparameter_tuning")
            
            # Grid search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log all parameter combinations
            for i, (params, score) in enumerate(zip(grid_search.cv_results_['params'], 
                                                   grid_search.cv_results_['mean_test_score'])):
                with mlflow.start_run(nested=True, run_name=f"combination_{i}"):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_score", score)
            
            self.best_params = grid_search.best_params_
            return grid_search.best_estimator_
    
    def train_final_model(self, model=None):
        """Train final model with best parameters"""
        if model is None:
            model = RandomForestClassifier(**self.best_params, random_state=42)
        
        with mlflow.start_run(run_name="final_model_training"):
            # Set tags
            mlflow.set_tag("stage", "production_training")
            mlflow.set_tag("model_type", "RandomForestClassifier")
            mlflow.set_tag("data_version", "v1.0")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                "train_accuracy": accuracy_score(self.y_train, y_pred_train),
                "test_accuracy": accuracy_score(self.y_test, y_pred_test),
                "precision": precision_score(self.y_test, y_pred_test),
                "recall": recall_score(self.y_test, y_pred_test),
                "f1_score": f1_score(self.y_test, y_pred_test)
            }
            
            # Log parameters and metrics
            mlflow.log_params(self.best_params)
            mlflow.log_metrics(metrics)
            
            # Create and log artifacts
            self.create_artifacts(model, y_pred_test, y_pred_proba)
            
            # Log model with signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(self.X_train_scaled, y_pred_train)
            
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name="wine_quality_classifier"
            )
            
            # Log preprocessing pipeline
            joblib.dump(self.scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
            
            return model
    
    def create_artifacts(self, model, y_pred, y_pred_proba):
        """Create and log various artifacts"""
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), y='feature', x='importance')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        # Save feature importance as CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt
