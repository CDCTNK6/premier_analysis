# Databricks notebook source
# Set up Azure storage connection
spark.conf.set("fs.azure.account.auth.type.davsynapseanalyticsdev.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.davsynapseanalyticsdev.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-tenant-id-endpoint"))

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

import argparse
import os
import pickle as pkl
from importlib import reload

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn

import tools.analysis as ta
import tools.preprocessing as tp

from databricks import feature_store 
from delta.tables import * 
from tools import preprocessing as tp

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# set experiment name
# mlflow.create_experiment("/Users/tnk6@cdc.gov/Test/E2E_tutorial")
experiment = mlflow.set_experiment("/Users/tnk6@cdc.gov/Test/E2E_premier")

print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Stage: {experiment.lifecycle_stage}")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Read delta table
# MAGIC - Note: TODO: compare delta table with current version and only merge if there are changes (?)

# COMMAND ----------

test_name = "tnk6_demo.output_test_dataset"
test_version = DeltaTable.forName(spark, test_name).history(1).collect().__getitem__(0)[0]
test_df = spark.table(test_name).toPandas()
test_df

# COMMAND ----------

train_name = "tnk6_demo.output_train_dataset"
train_version = DeltaTable.forName(spark, train_name).history(1).collect().__getitem__(0)[0]
train_df = spark.table(train_name).toPandas()

# COMMAND ----------

val_name = "tnk6_demo.output_val_dataset"
val_version = DeltaTable.forName(spark, val_name).history(1).collect().__getitem__(0)[0]
val_df = spark.table(val_name).toPandas()

# COMMAND ----------

X_test = test_df['features'].apply(pd.Series)
X_train = train_df['features'].apply(pd.Series)
X_val = val_df['features'].apply(pd.Series)
y_test = test_df['target']
y_train = train_df['target']
y_val = val_df['target']

# COMMAND ----------

print(sum(y_test)/len(y_test))
print(sum(y_train)/len(y_train))
print(sum(y_val)/len(y_val))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Drift Detection

# COMMAND ----------

val_df['features'].apply(lambda x: (x<0).any()).any()

# COMMAND ----------

def check_validity(df):
    if df['features'].apply(lambda x: (x<0).any()).any():
        return "invalid"
    else:
        return "valid"

# COMMAND ----------

check_validity(val_df)

# COMMAND ----------

error_df = val_df.head()
error_df['features'] = [x - 1 for x in error_df['features']]
check_validity(error_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Building Baseline Model

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='E2E_untuned_random_forest'):
  n_estimators = 200
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

roc_auc_score(y_test[0:100], model.predict_proba(X_test[0:100])[:,1])

# COMMAND ----------

# Register model in MLFlow Registry
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "E2E_untuned_random_forest"').iloc[0].run_id
print(run_id)

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = "premier_ml_E2E"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition model to production

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md # Hyperparameter Tuning

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)
    # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=50,
    trials=spark_trials,
  )

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Batch Inference

# COMMAND ----------

import mlflow.pyfunc
model_name = ''
print(model_name)
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)
