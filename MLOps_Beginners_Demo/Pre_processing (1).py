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

# MAGIC %md ## Import data
# MAGIC   
# MAGIC In this section, you download a dataset from the web and upload it to Databricks File System (DBFS).
# MAGIC 
# MAGIC 1. Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both `winequality-red.csv` and `winequality-white.csv` to your local machine.
# MAGIC 
# MAGIC 1. From this Databricks notebook, select *File* > *Upload Data*, and drag these files to the drag-and-drop target to upload them to the Databricks File System (DBFS). 
# MAGIC 
# MAGIC     **Note**: if you don't have the *File* > *Upload Data* option, you can load the dataset from the Databricks example datasets. Uncomment and run the last two lines in the following cell.
# MAGIC 
# MAGIC 1. Click *Next*. Some auto-generated code to load the data appears. Select *pandas*, and copy the example code. 
# MAGIC 
# MAGIC 1. Create a new cell, then paste in the sample code. It will look similar to the code shown in the following cell. Make these changes:
# MAGIC   - Pass `sep=';'` to `pd.read_csv`
# MAGIC   - Change the variable names from `df1` and `df2` to `white_wine` and `red_wine`, as shown in the following cell.

# COMMAND ----------

# If you have the File > Upload Data menu option, follow the instructions in the previous cell to upload the data from your local machine.
# The generated code, including the required edits described in the previous cell, is shown here for reference.

import pandas as pd

# In the following lines, replace <username> with your username.
white_wine = spark.table("mlops_july_demo.raw_winequality_white")
white_wine = white_wine.toPandas()
red_wine = spark.table("mlops_july_demo.raw_winequality_white")
red_wine = red_wine.toPandas()

white_wine

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## Visualize data
# MAGIC 
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC 
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data['quality'] = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC 
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataset for training baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

# MAGIC %md 
# MAGIC ```
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC 
# MAGIC X = data.drop(["quality"], axis=1)
# MAGIC y = data.quality
# MAGIC 
# MAGIC # Split out the training data
# MAGIC X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
# MAGIC 
# MAGIC # Split the remaining data equally into validation and test
# MAGIC X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)
# MAGIC ```

# COMMAND ----------

from sklearn.model_selection import train_test_split


# Split out the training data
data_train, data_rem = train_test_split(data, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
data_val, data_test = train_test_split(data_rem, test_size=0.5, random_state=123)

# COMMAND ----------

data_val

# COMMAND ----------

def save_dataset_asDelta(df, table_name):
  df = spark.createDataFrame(df)
  df.write.format("delta").saveAsTable(table_name)

# COMMAND ----------

save_dataset_asDelta(data_train, "mlops_july_demo.int_train_dataset")
save_dataset_asDelta(data_val, "mlops_july_demo.int_test_dataset")
save_dataset_asDelta(data_test, "mlops_july_demo.int_val_dataset")
