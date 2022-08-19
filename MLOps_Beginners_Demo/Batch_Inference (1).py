# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: an end-to-end example
# MAGIC 
# MAGIC This tutorial covers the following steps:
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train machine learning models on the dataset
# MAGIC - Explore the results of the hyperparameter sweep with MLflow
# MAGIC - Register the best performing model in MLflow
# MAGIC - Apply the registered model to another dataset using a Spark UDF
# MAGIC - Set up model serving for low-latency requests
# MAGIC 
# MAGIC In this example, you build a model to predict the quality of Portugese "Vinho Verde" wine based on the wine's physicochemical properties. 
# MAGIC 
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC 
# MAGIC ## Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning.  
# MAGIC If you are using Databricks Runtime 7.3 LTS ML or below, you must update the CloudPickle library. To do that, uncomment and run the `%pip install` command in Cmd 2. 
# MAGIC 
# MAGIC ## Source
# MAGIC https://docs.databricks.com/delta/quick-start.html#read-a-table

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------


data_test = spark.table("mlops_july_demo.int_test_dataset").toPandas()
y_test = data_test.pop('quality')
X_test = data_test

# COMMAND ----------

# MAGIC %md The auc value on the test set for the new model is 0.90. You beat the baseline!

# COMMAND ----------

# MAGIC %md ##Batch inference
# MAGIC 
# MAGIC There are many scenarios where you might want to evaluate a model on a corpus of new data. For example, you may have a fresh batch of data, or may need to compare the performance of two models on the same corpus of data.
# MAGIC 
# MAGIC The following code evaluates the model on data stored in a Delta table, using Spark to run the computation in parallel.

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_test)
# Replace <username> with your username before running this cell.
table_path = "dbfs:/<username>/delta/wine_data"
# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md Load the model into a Spark UDF, so it can be applied to the Delta table.

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)
