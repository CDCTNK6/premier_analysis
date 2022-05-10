# Databricks notebook source
# need to refactor to add flags 
## %run feature_extraction.py 
# %run 01_feature_extraction

# COMMAND ----------

# MAGIC %run ./feature_tokenization_notebook   # note: lag in creation of file and running; also: naming must not start with numbers

# COMMAND ----------

##%run /Workspace/Repos/tnk6@cdc.gov/premier_analysis/python/refactored_feature_extraction.py
%run refactored_feature_extraction.py

# COMMAND ----------

##%run /Workspace/Repos/tnk6@cdc.gov/premier_analysis/python/refactored_feature_tokenization.py
%run refactored_feature_tokenization.py

# COMMAND ----------

# MAGIC %ls

# COMMAND ----------

##%run 
%run sequence_trimming.py --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

#
%run baseline_models.py --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/' --outcome='icu'

# COMMAND ----------

dbutils.fs.mkdirs('/home/tnk6/model_checkpoints/')

# COMMAND ----------

# MAGIC %pip install keras-tuner

# COMMAND ----------

# Run DAN
%run model.py --outcome='icu' --day_one --model='dan' --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------

dbutils.fs.mkdirs('/home/tnk6/model_checkpoints/best/dan')

# COMMAND ----------

#
%run model.py --outcome='icu' --day_one --model='hp_dan' --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------


