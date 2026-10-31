import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_Tip_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema
tip_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("business_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("date", StringType(), True),
    StructField("compliment_count", IntegerType(), True)
])

print("--> Reading tip.json with explicit schema...")
df_tip = spark.read.schema(tip_schema).json(f"{raw_dir}/yelp_academic_dataset_tip.json")

print("--> Writing to Parquet format...")
df_tip.write.mode("overwrite").parquet(f"{out_dir}/yelp_tip.parquet")

print("--> Finished successfully!")
spark.stop()
