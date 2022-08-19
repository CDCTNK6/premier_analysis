# Databricks notebook source
# MAGIC %pip install keras-tuner --quiet
# MAGIC ##%pip install cudf --quiet

# COMMAND ----------

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
import csv
import os
import pickle as pkl

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

import tools.analysis as ta
import tools.preprocessing as tp
import tools.keras as tk

import mlflow
import mlflow.tensorflow
import mlflow.keras
import datetime, uuid

#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import kerastuner

# COMMAND ----------


from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

import mlflow

# create experiment 
# mlflow.create_experiment("/Users/tnk6@cdc.gov/Test/E2E_premier_DAN")
# set experiment name
experiment = mlflow.set_experiment("/Users/tnk6@cdc.gov/Test/E2E_premier_DAN")

print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Stage: {experiment.lifecycle_stage}")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Read delta table
# MAGIC - Note: TODO: compare delta table with current version and only merge if there are changes (?)

# COMMAND ----------

from delta.tables import *
test_name = "tnk6_premier_demo.output_test_dataset"
test_version = DeltaTable.forName(spark, test_name).history(1).collect().__getitem__(0)[0]
test_df = spark.table(test_name) #.toPandas()
#display(test_df)

# COMMAND ----------

import pyspark.sql.functions as F
# testing
df = spark.createDataFrame([(["c", "b", "a","e","a"],)], ['arraydata'])
user_func = udf (lambda x,y: [i+1 for i, e in enumerate(x) if e==y ])
newdf = df.withColumn('item_position',user_func(df.arraydata,F.lit('a')))
newdf.show()

# COMMAND ----------

test_df = test_df.withColumn("feature_list", user_func(F.col("features"),F.lit(1)))
test_df = test_df[['target','feature_list']].toPandas()
test_df

# COMMAND ----------

train_name = "tnk6_premier_demo.output_train_dataset"
train_version = DeltaTable.forName(spark, train_name).history(1).collect().__getitem__(0)[0]
train_df = spark.table(train_name) #.toPandas()
train_df = train_df.withColumn("feature_list", user_func(F.col("features"),F.lit(1)))
train_df = train_df[['target','feature_list']].toPandas()

# COMMAND ----------

train_df

# COMMAND ----------

val_name = "tnk6_premier_demo.output_val_dataset"
val_version = DeltaTable.forName(spark, val_name).history(1).collect().__getitem__(0)[0]
val_df = spark.table(val_name) #.toPandas()
val_df = val_df.withColumn("feature_list", user_func(F.col("features"),F.lit(1)))
val_df = val_df[['target','feature_list']].toPandas()

# COMMAND ----------

# very slow using pandas
# val_df['features'].apply(lambda x: ["c"+str(i) for i, y in enumerate(x) if y == 1])

# COMMAND ----------

#elem] for elem in val_df['feature_list']]
import ast 
#features = [x for x in val_df['feature_list'].tolist()]
features = val_df['feature_list'].tolist()
print(type(features[0]))
#list(features[0].astype(str).str.split(";"))
ast.literal_eval(features[0])

# COMMAND ----------

val_df

# COMMAND ----------

val_df['feature_list'] = val_df['feature_list'].apply(lambda x: ast.literal_eval(x))
features = [[int(elem) for elem in lstt[:]] for lstt in val_df['feature_list'].tolist()]
X_val = keras.preprocessing.sequence.pad_sequences(features,maxlen=300,
                                                   padding='post')
X_val

# COMMAND ----------

train_df['feature_list'] = train_df['feature_list'].apply(lambda x: ast.literal_eval(x))
features = [[int(elem) for elem in lstt[:]] for lstt in train_df['feature_list'].tolist()]
X_train = keras.preprocessing.sequence.pad_sequences(features,maxlen=300,
                                                   padding='post')
test_df['feature_list'] = test_df['feature_list'].apply(lambda x: ast.literal_eval(x))
features = [[int(elem) for elem in lstt[:]] for lstt in test_df['feature_list'].tolist()]
X_test = keras.preprocessing.sequence.pad_sequences(features,maxlen=300,
                                                   padding='post')

# COMMAND ----------

#X_test = test_df['features'].apply(pd.Series)
#X_test = test_df['features'].apply(lambda x: ["c"+str(i) for i, y in enumerate(x) if y == 1])
#X_train = train_df['features'].apply(pd.Series)
#X_test = train_df['features'].apply(lambda x: ["c"+str(i) for i, y in enumerate(x) if y == 1])
#X_val = val_df['features'].apply(pd.Series)
#X_val = val_df['features'].apply(lambda x: ["c"+str(i) for i, y in enumerate(x) if y == 1])
y_test = test_df['target']
y_train = train_df['target']
y_val = val_df['target']

# COMMAND ----------

print(sum(y_test)/len(y_test))
print(sum(y_train)/len(y_train))
print(sum(y_val)/len(y_val))

# COMMAND ----------

N_VOCAB = np.max(train_df['feature_list'].apply(lambda x: max(x)))
TIME_SEQ = X_train.shape[1]
TIME_SEQ

# COMMAND ----------

# MAGIC %md 
# MAGIC # Keras Prep

# COMMAND ----------

metrics = [
    keras.metrics.AUC(num_thresholds=int(1e5), name="ROC_AUC"),
   keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR_AUC"),
]

try:
    username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
    username = str(uuid.uuid1()).replace("-","")
experiment_log_dir = f"/dbfs/user/{username}/tensorboard_log_dir"
print(experiment_log_dir)

##%load_ext tensorboard
##%tensorboard --logdir $experiment_log_dir

# COMMAND ----------

tensorboard_dir = experiment_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(tensorboard_dir)
### REPLACE WITH MLFLOW
# Create some callbacks
callbacks = [
    TensorBoard(
        #log_dir=os.path.join(tensorboard_dir, OUTCOME, MOD_NAME),
        log_dir=tensorboard_dir,
        histogram_freq=1) #,
        #update_freq=TB_UPDATE_FREQ,
        #embeddings_freq=5,
        #embeddings_metadata=os.path.join(tensorboard_dir,
        #                                 "emb_metadata.tsv"),
    #)
    ,

    # Create model checkpoint callback
    #keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    #    tensorboard_dir, OUTCOME, MOD_NAME,
    #    "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
    #                                save_weights_only=True,
    #                                monitor="val_loss",
    #                                mode="max",
    #                                save_best_only=True),
    #
    # Create early stopping callback
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  min_delta=0,
                                  patience=20,
                                  mode="auto")
]

# COMMAND ----------

# MAGIC %md 
# MAGIC # Building Baseline Model

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

loss_fn = keras.losses.binary_crossentropy
BATCH_SIZE=64
print('Starting DAN model')


with mlflow.start_run(run_name='test_Dan') as run:
    mlflow.tensorflow.autolog()
    
    mlflow.log_param("Test Dataset",test_name)
    mlflow.log_param("Test Dataset Version",test_version )
    mlflow.log_param("Train Dataset",train_name)
    mlflow.log_param("Train Dataset Version",train_version )
    mlflow.log_param("Validation Dataset",val_name)
    mlflow.log_param("Validation Dataset Version",val_version )
    model = tk.DAN(vocab_size=N_VOCAB,
                   ragged=False,
                   input_length=TIME_SEQ, embedding_size=16)

    model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

    # train the model 
    results = model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE,
              epochs=4,
              validation_data=(X_val, y_val),
              callbacks=callbacks)
              #class_weight=weight_dict)

    # Produce DAN predictions on validation and test sets
    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)
    mlflow.keras.log_model(model,"DAN")

# Use mlflow to log the run and model 
#with mlflow.start_run(run_name='test_Dan') as run:
#    mlflow.keras.log_model(model,"DAN")


# COMMAND ----------

mlflow.search_runs(filter_string='tags.mlflow.runName = "test_Dan"')

# COMMAND ----------

# Register model in MLFlow Registry
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "test_Dan"').iloc[0].run_id
print(run_id)

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = "premier_dan_E2E"
model_version = mlflow.register_model(f"runs:/{run_id}/DAN", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# Test model
import mlflow
logged_model = f'runs:/{run_id}/DAN'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
predictions = loaded_model.predict(X_test)
predictions

# COMMAND ----------

print(len(predictions))
print(len(y_test))
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, predictions)

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

print(X_test.shape)
print(X_train.shape)
X_test

# COMMAND ----------

# MAGIC %md # Hyperparameter Tuning

# COMMAND ----------


N_FEATS = X_train.shape[1]
hypermodel = tk.DANHyper(
    vocab_size = N_VOCAB,
    input_size = N_FEATS,
    metrics = metrics,
    n_classes = 2)

# COMMAND ----------

tensorboard_dir = experiment_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(tensorboard_dir)
### REPLACE WITH MLFLOW
# Create some callbacks
callbacks = [
#    TensorBoard(
        #log_dir=os.path.join(tensorboard_dir, OUTCOME, MOD_NAME),
#        log_dir=tensorboard_dir,
#        histogram_freq=1) #,
        #update_freq=TB_UPDATE_FREQ,
        #embeddings_freq=5,
        #embeddings_metadata=os.path.join(tensorboard_dir,
        #                                 "emb_metadata.tsv"),
    #)
#    ,
#
    # Create model checkpoint callback
    #keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    #    tensorboard_dir, OUTCOME, MOD_NAME,
    #    "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
    #                                save_weights_only=True,
    #                                monitor="val_loss",
    #                                mode="max",
    #                                save_best_only=True),
    #
    # Create early stopping callback
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  min_delta=0,
                                  patience=4,
                                  mode="auto")
]

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

EPOCHS=20
with mlflow.start_run(run_name="DAN_tuning", nested=True):
    mlflow.tensorflow.autolog(log_models=True)
    mlflow.log_param("Test Dataset",test_name)
    mlflow.log_param("Test Dataset Version",test_version )
    mlflow.log_param("Train Dataset",train_name)
    mlflow.log_param("Train Dataset Version",train_version )
    mlflow.log_param("Validation Dataset",val_name)
    mlflow.log_param("Validation Dataset Version",val_version )
    #with mlflow.start_run(run_name="DAN_tuning", nested=True):
    tuner = kerastuner.tuners.bayesian.BayesianOptimization(
        hypermodel,
        max_trials = 10,
        objective= "val_loss",
        #project_name = "dan_hp_tune",
        directory=tensorboard_dir)
    print(tuner.search_space_summary())
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs= EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
mlflow.end_run()
    #mlflow.keras.log_model(some_ddp_model, some_path, pip_requirements=[f"torch=={torch.__version__}"])

# COMMAND ----------

# get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0].values

print(best_hps)
#print(best_hps.get('units'))
#print(best_hps.get('learning_rate'))

# COMMAND ----------

tuner.get_best_models()[0].summary()

# COMMAND ----------

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
#mlflow.tensorflow.autolog()
with mlflow.start_run(run_name="DAN_best_model"):
    mlflow.tensorflow.autolog(log_models=True)
    mlflow.log_param("Test Dataset",test_name)
    mlflow.log_param("Test Dataset Version",test_version )
    mlflow.log_param("Train Dataset",train_name)
    mlflow.log_param("Train Dataset Version",train_version )
    mlflow.log_param("Validation Dataset",val_name)
    mlflow.log_param("Validation Dataset Version",val_version )
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    history = model.fit(X_train,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=200,
          validation_data=(X_val, y_val),
          callbacks=callbacks) #,
#          class_weight=weight_dict)
# mlflow.keras.log_model(model, "dan")
mlflow.end_run()
val_probs = model.predict(X_val)
test_probs = model.predict(X_test)

# Use mlflow to log the run and model 
#with mlflow.start_run(run_name='test_Dan') as run:
#    mlflow.keras.log_model(model,"DAN")

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.val_ROC_AUC DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.val_ROC_AUC"]}')

# COMMAND ----------

mlflow.search_runs().columns

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)
print(new_model_version)
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
