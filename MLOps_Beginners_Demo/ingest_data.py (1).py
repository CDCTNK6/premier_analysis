# Databricks notebook source
# MAGIC %md 
# MAGIC # Ingesting Data
# MAGIC ### Assumptions:
# MAGIC Health data is available through CDH as delta tables
# MAGIC 
# MAGIC - This script shouldn't be used. It is just to demonstrate where the data for the demo is comming from

# COMMAND ----------

# Import libraries
import pandas as pd

# COMMAND ----------

# Read data from external source
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# Create a database in Databricks 
spark.sql("CREATE DATABASE IF NOT EXISTS mlops_july_demo")

# COMMAND ----------

# Convert pandas dataframe to spark data frame 
# Pandas cannot be saved as delta tables
white_wine_spark = spark.createDataFrame(white_wine) 
red_wine_spark = spark.createDataFrame(red_wine) 

# COMMAND ----------

white_wine_spark = white_wine_spark.toDF(*[name.replace(' ','_') for name in white_wine_spark.columns])
red_wine_spark = red_wine_spark.toDF(*[name.replace(' ','_') for name in red_wine_spark.columns])

# COMMAND ----------

# Check content of spark data frame
display(white_wine_spark)

# COMMAND ----------

# Print the schema
white_wine_spark.printSchema()

# COMMAND ----------

# Save as delta tables
white_wine_spark.write.format("delta").saveAsTable("mlops_july_demo.raw_winequality_white")
red_wine_spark.write.format("delta").saveAsTable("mlops_july_demo.raw_winequality_red")

# COMMAND ----------


