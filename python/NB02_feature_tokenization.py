# Databricks notebook source
# Set up Azure storage connection
spark.conf.set("fs.azure.account.auth.type.davsynapseanalyticsdev.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.davsynapseanalyticsdev.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-tenant-id-endpoint"))

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.feature_extraction.text import CountVectorizer
from tools import preprocessing as tp
from databricks import feature_store
from pyspark.ml.feature import CountVectorizer as sparkCV

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS tnk6_demo

# COMMAND ----------



# Enable Arrow-based columnar data transfers
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# %% Setting top-level parameters
# Note: refactored to read files from delta tables
MIN_DF = 5
NO_VITALS = False
ADD_DEMOG = True
TIME_UNIT = "dfi"
REVERSE_VOCAB = True
MISA_ONLY = True

# Whether to write the full trimmed sequence file to disk as pqruet
WRITE_PARQUET = True

# Setting the directories
prem_dir = '/dbfs/home/tnk6/premier/'
output_dir = '/dbfs/home/tnk6/premier_output/'
 
data_dir = prem_dir
# Note: used git traverse to put targets in dbfs
targets_dir = os.path.join(output_dir, "targets", "")
pkl_dir = os.path.join(output_dir, "pkl", "")

ftr_cols = ['vitals', 'bill', 'genlab', 'lab_res', 'proc', 'diag']
demog_vars = ["gender", "hispanic_ind", "age", "race"]
final_cols = ['covid_visit', 'ftrs']

# COMMAND ----------

comp_pat_df = pd.read_parquet(data_dir + "vw_covid_pat_all/")
comp_id_df = pd.read_parquet(data_dir + "vw_covid_id/")
comp_provider = pd.read_parquet(data_dir + "providers/")


# COMMAND ----------

comp_pat_df

# COMMAND ----------

comp_id_df

# COMMAND ----------

comp_provider

# COMMAND ----------

# Note: refactored to read files from delta tables
# %% Read in the pat and ID tables
#pat_df = pd.read_parquet(data_dir + "vw_covid_pat_all/")
#id_df = pd.read_parquet(data_dir + "vw_covid_id/")
#provider = pd.read_parquet(data_dir + "providers/")
#misa_data = pd.read_csv(targets_dir + 'icu_targets.csv')

pat_df = tp.read_table(data_dir,"vw_covid_pat_all")  
id_df = tp.read_table(data_dir, "vw_covid_id")
provider = tp.read_table(data_dir,"providers")
# TO DO: (pre-process targets - see Github premier_data)
misa_data = pd.read_csv(targets_dir + 'icu_targets.csv')
#misa_data = misa_data.head(10000) # use for testing 

# COMMAND ----------

pat_df

# COMMAND ----------

id_df

# COMMAND ----------

provider

# COMMAND ----------

print(comp_provider.columns)
print(provider.columns)
print(id_df.columns)
print(comp_id_df.columns)
print(pat_df.columns)
print(comp_pat_df.columns)
[set(comp_id_df) - set(id_df)]

# COMMAND ----------

misa_data

# COMMAND ----------

comp_trimmed_seq = pd.read_parquet(output_dir + "parquet/flat_features.parquet")
comp_trimmed_seq

# COMMAND ----------

# Read in the flat feature file (output of NB01_feature_extraction)
#trimmed_seq = pd.read_parquet(output_dir + "parquet/flat_features.parquet")
trimmed_seq = tp.read_table(data_dir,"intertim_flat_features")
trimmed_seq

# COMMAND ----------

# %% Filter Denom to those identified in MISA case def
print(MISA_ONLY)
if MISA_ONLY:
    trimmed_seq = trimmed_seq[trimmed_seq.medrec_key.isin(
        misa_data.medrec_key)]

# Determine unique patients
n_patients = trimmed_seq["medrec_key"].nunique()
print(f'Number of patients: {n_patients}')
# Ensure we're sorted

trimmed_seq.sort_values(["medrec_key", "dfi"], inplace=True)

# %% Optionally drops vitals and genlab from the features
print(NO_VITALS)
if NO_VITALS:
    ftr_cols = ['bill', 'lab_res', 'proc', 'diag']

# Combining the separate feature columns into one
trimmed_seq["ftrs"] = (trimmed_seq[ftr_cols].astype(str).replace(
    ["None", "nan"], "").agg(" ".join, axis=1))
trimmed_seq

# COMMAND ----------

# MAGIC %md 
# MAGIC # Using Spark CountVectorizer

# COMMAND ----------

# Using Spark 
from pyspark.sql.functions import array
tmp_df = trimmed_seq[['medrec_key','pat_key','ftrs']]
tmp_df = spark.createDataFrame(tmp_df)
display(tmp_df)

# COMMAND ----------

import pyspark.sql.functions as F
#tmp_df = tmp_df.withColumn('ftrs', array(tmp_df['ftrs']))
#tmp_df = tmp_df.select(F.split(F.col("ftrs")," ").alias("ftrs_array")).drop("ftrs")
tmp_df = tmp_df.withColumn('ftrs', F.split(F.col("ftrs")," "))
display(tmp_df)

# COMMAND ----------

sp_cv = sparkCV()
sp_cv.setInputCol('ftrs')
sp_cv.setOutputCol('ftrs_vectors')
sp_model = sp_cv.fit(tmp_df)
sp_model.setInputCol('ftrs')
ftrs_df = sp_model.transform(tmp_df) #.show(truncate=False)
display(ftrs_df)

# COMMAND ----------

#sp_model.show(truncate=True)
sorted(sp_model.vocabulary)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using original vectorization

# COMMAND ----------

trimmed_seq.ftrs

# COMMAND ----------

# %% Fitting the vectorizer to the features
ftrs = [doc for doc in trimmed_seq.ftrs]
vec = CountVectorizer(ngram_range=(1, 1), min_df=MIN_DF, binary=True)
vec.fit(ftrs)
vocab = vec.vocabulary_

# Saving the index 0 for padding
for k in vocab.keys():
    vocab[k] += 1

vocab

# COMMAND ----------

# Converting the bags of feature strings to integers
int_ftrs = [[vocab[k] for k in doc.split() if k in vocab.keys()]
            for doc in ftrs]
trimmed_seq["int_ftrs"] = int_ftrs
trimmed_seq

# COMMAND ----------

# list of integer sequence arrays split by medrec_key
# groups by medrec and puts the values into an list of array of lists
int_seqs = [
    df.values for _, df in trimmed_seq.groupby("medrec_key")["int_ftrs"]
]
int_seqs

# COMMAND ----------

# Converting to a nested list to keep things clean
seq_gen = [[seq for seq in medrec] for medrec in int_seqs]
seq_gen

# COMMAND ----------

trimmed_seq

# COMMAND ----------

print(demog_vars)
pat_df

# COMMAND ----------

print(ADD_DEMOG)

# COMMAND ----------

# %% Optionally add demographics
if ADD_DEMOG:
    # Append demog
    trimmed_plus_demog = trimmed_seq.merge(pat_df[["medrec_key"] + demog_vars],
                                           how="left").set_index("medrec_key")

    if "age" in demog_vars:
        trimmed_plus_demog = tp.max_age_bins(trimmed_plus_demog,
                                             bins=np.arange(0, 111, 10))

    # %% Take distinct by medrec
    demog_map = map(lambda name: name + ":" + trimmed_plus_demog[name],
                    demog_vars)
    demog_labeled = pd.concat(demog_map, axis=1)
    raw_demog = demog_labeled.reset_index().drop_duplicates()
    just_demog = raw_demog.groupby("medrec_key").agg(
        lambda x: " ".join(list(set(x))).lower())

    # BUG: Note there are some medrecs with both hispanic=y and hispanic=N
    just_demog["all_demog"] = just_demog[demog_vars].agg(" ".join, axis=1)
    demog_list = [demog for demog in just_demog.all_demog]
    assert just_demog.shape[0] == n_patients, "No funny business"
    demog_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[\w:-]+\b")
    demog_vec.fit(demog_list)
    demog_vocab = demog_vec.vocabulary_
    # This allows us to use 0 for padding if we coerce to dense
    for k in demog_vocab.keys():
        demog_vocab[k] += 1
    demog_ints = [[
        demog_vocab[k] for k in doc.split() if k in demog_vocab.keys()
    ] for doc in demog_list]

    # Zip with seq_gen to produce a list of tuples
    seq_gen = [seq for seq in zip(seq_gen, demog_ints)]

    # And saving vocab
#    with open(pkl_dir + "demog_dict.pkl", "wb") as f:
#        pkl.dump(demog_vocab, f)

# COMMAND ----------

demog_ints

# COMMAND ----------

demog_vocab

# COMMAND ----------

print(seq_gen[0][0])
print('----------')
print(seq_gen[0][1])

# COMMAND ----------

# === Figuring out which visits were covid visits,
# and which patients have no covid visits (post-trim)

cv_dict = dict(zip(pat_df.pat_key, pat_df.covid_visit))
cv_pats = [[cv_dict[pat_key] for pat_key in np.unique(seq.values)]
           for _, seq in trimmed_seq.groupby("medrec_key").pat_key]
cv_pats

# COMMAND ----------

# FIX?: Remove no_covid indices
print(len(cv_pats))
no_covid = np.where([np.sum(doc) == 0 for doc in cv_pats])[0]
print(len(no_covid))
cv_pats = [e for i,e in enumerate(cv_pats) if i not in no_covid]
print(len(cv_pats))
cv_pats

# COMMAND ----------

# With the new trimming, this should never be populated
# Note commenting out as new data fails the assertion
assert len(no_covid) == 0

# Additional sanity check
assert len(cv_pats) == len(seq_gen) == trimmed_seq.medrec_key.nunique()

trimmed_seq

# COMMAND ----------

trimmed_seq

# COMMAND ----------

# Writing the trimmed sequences to disk
if WRITE_PARQUET:
    #trimmed_seq.to_parquet(output_dir + 'parquet/trimmed_seq.parquet')
    tmp_to_save = spark.createDataFrame(trimmed_seq)
    tmp_to_save.write.mode("overwrite").format("delta").saveAsTable("tnk6_demo.interim_trimmed_seq")

# COMMAND ----------

# Save list-of-list-of-lists as pickle
with open(pkl_dir + "int_seqs_fromdelta.pkl", "wb") as f:
    pkl.dump(seq_gen, f)

# COMMAND ----------

# MAGIC %sql 
# MAGIC --drop table tnk6_demo.interim_int_seqs_pkl

# COMMAND ----------

tmp_to_save = pd.DataFrame(seq_gen)
tmp_to_save = spark.createDataFrame(tmp_to_save)
display(tmp_to_save)

# COMMAND ----------

tmp_to_save.write.mode("overwrite").format("delta").saveAsTable("tnk6_demo.interim_int_seqs_pkl")

# COMMAND ----------

# MAGIC %sql
# MAGIC --DROP TABLE tnk6_demo.interim_int_seqs_pkl

# COMMAND ----------

# Freeing up memory
seq_gen = []

# Figuring out how many feature bags in each sequence belong
# to each visit
pat_lengths = trimmed_seq.groupby(["medrec_key", "pat_key"],
                                  sort=False).pat_key.count()
pat_lengths = [[n for n in df.values]
               for _, df in pat_lengths.groupby("medrec_key")]

# %% Making a groupby frame to use below
grouped_pat_keys = trimmed_seq.groupby("medrec_key").pat_key

# %% Figuring out whether a patient died after a visit
died = np.array(["EXPIRED" in status for status in pat_df.disc_status_desc],
                dtype=np.uint8)
death_dict = dict(zip(pat_df.pat_key, died))
pat_deaths = [[death_dict[id] for id in np.unique(df.values)]
              for _, df in grouped_pat_keys]

# Adding the inpatient variable to the pat dict
inpat = np.array(pat_df.pat_type == 8, dtype=np.uint8)
inpat_dict = dict(zip(pat_df.pat_key, inpat))
pat_inpat = [[inpat_dict[id] for id in np.unique(df.values)]
             for _, df in grouped_pat_keys]

# %% Adding the ICU indicator
icu_pats = misa_data[misa_data.icu_visit == "Y"].pat_key
icu_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
for pat in icu_pats:
    icu_dict.update({pat: 1})
icu = [[icu_dict[id] for id in np.unique(df.values)]
       for _, df in grouped_pat_keys]

# %% Adding age at each visit
age = pat_df.age.values.astype(np.uint8)
age_dict = dict(zip(pat_df.pat_key, age))
pat_age = [[age_dict[id] for id in np.unique(df.values)]
           for _, df in grouped_pat_keys]

# Mixing in the MIS-A targets and Making a lookup for the first case definition
misa_pt_pats = misa_data[misa_data.misa_filled == 1].pat_key
misa_pt_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
for pat in misa_pt_pats:
    misa_pt_dict.update({pat: 1})

misa_pt = [[misa_pt_dict[id] for id in np.unique(df.values)]
           for _, df in grouped_pat_keys]

#  And finally saving a the pat_keys themselves to facilitate
# record linkage during analysis
pat_key = [[num for num in df.values] for _, df in grouped_pat_keys]

# Rolling things up into a dict for easier saving
pat_dict = {
    'key': pat_key,
    'age': pat_age,
    'covid': cv_pats,
    'length': pat_lengths,
    'inpat': pat_inpat,
    'outcome': {
        'icu': icu,
        'death': pat_deaths,
        'misa_pt': misa_pt
    }
}

# COMMAND ----------

pat_dict['outcome']['icu']

# COMMAND ----------

pd.DataFrame.from_dict(pat_dict,orient='index')

# COMMAND ----------

# %%
with open(pkl_dir + "pat_data_fromdelta.pkl", "wb") as f:
    pkl.dump(pat_dict, f)


# COMMAND ----------

display(tmp_to_save)

# COMMAND ----------

# MAGIC %sql 
# MAGIC --DROP TABLE tnk6_demo.all_ftrs_dict_pkl

# COMMAND ----------

# Optionally reversing the vocab
if REVERSE_VOCAB:
    vocab = {v: k for k, v in vocab.items()}

# Saving the updated vocab to disk
#with open(pkl_dir + "all_ftrs_dict.pkl", "wb") as f:
#    pkl.dump(vocab, f)

#tmp_to_save = pd.DataFrame.from_dict(vocab,orient='index',columns=['value'])
tmp_to_save = pd.DataFrame(vocab.items(),columns=['index','value'])
tmp_to_save = spark.createDataFrame(tmp_to_save)


# COMMAND ----------

display(tmp_to_save)

# COMMAND ----------

#tmp_to_save.write.mode("ignore").format("delta").saveAsTable("tnk6_demo.all_ftrs_dict_pkl")
from pyspark.sql.types import StringType, FloatType, LongType
tmp_to_save = tmp_to_save.withColumn("index", tmp_to_save["index"].cast(LongType()))
tmp_to_save.write.mode("overwrite").format("delta").saveAsTable("tnk6_demo.all_ftrs_dict_pkl")

# COMMAND ----------

# MAGIC %sql
# MAGIC --drop table tnk6_demo.all_ftrs_dict_pkl

# COMMAND ----------


