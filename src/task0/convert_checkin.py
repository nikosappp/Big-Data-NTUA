import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_Checkin_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema
checkin_schema = StructType([
    StructField("business_id", StringType(), True),
    StructField("date", StringType(), True)
])

print("--> Reading checkin.json with explicit schema...")
df_checkin = spark.read.schema(checkin_schema).json(f"{raw_dir}/yelp_academic_dataset_checkin.json")

print("--> Writing to Parquet format...")
df_checkin.write.mode("overwrite").parquet(f"{out_dir}/yelp_checkin.parquet")

print("--> Finished checkin conversion successfully!")
spark.stop()
