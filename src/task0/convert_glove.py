import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, slice, transform

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_GloVe_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

print("--> Reading glove.6B.100d.txt as text...")
df_raw = spark.read.text(f"{raw_dir}/glove.6B.100d.txt")

# Split line by whitespace
tokens = split(col("value"), " ")

# Extract word (1st element) and vector (remaining 100 elements cast to float)
df_glove = df_raw.select(
    tokens.getItem(0).alias("word"),
    transform(slice(tokens, 2, 100), lambda x: x.cast("float")).alias("vec")
)

print("--> Writing GloVe to Parquet with schema (word STRING, vec ARRAY)...")
df_glove.write.mode("overwrite").parquet(f"{out_dir}/glove_100d.parquet")

print("--> Finished GloVe conversion successfully!")
spark.stop()
