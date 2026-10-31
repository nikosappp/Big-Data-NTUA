import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, LongType
)

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_User_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema
user_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("review_count", IntegerType(), True),
    StructField("yelping_since", StringType(), True),
    StructField("useful", LongType(), True),
    StructField("funny", LongType(), True),
    StructField("cool", LongType(), True),
    StructField("elite", StringType(), True),
    StructField("friends", StringType(), True),
    StructField("fans", IntegerType(), True),
    StructField("average_stars", DoubleType(), True),
    StructField("compliment_hot", IntegerType(), True),
    StructField("compliment_more", IntegerType(), True),
    StructField("compliment_profile", IntegerType(), True),
    StructField("compliment_cute", IntegerType(), True),
    StructField("compliment_list", IntegerType(), True),
    StructField("compliment_note", IntegerType(), True),
    StructField("compliment_plain", IntegerType(), True),
    StructField("compliment_cool", IntegerType(), True),
    StructField("compliment_funny", IntegerType(), True),
    StructField("compliment_writer", IntegerType(), True),
    StructField("compliment_photos", IntegerType(), True)
])

print("--> Reading user.json with explicit schema...")
df_user = spark.read.schema(user_schema).json(f"{raw_dir}/yelp_academic_dataset_user.json")

print("--> Writing to Parquet format...")
df_user.write.mode("overwrite").parquet(f"{out_dir}/yelp_user.parquet")

print("--> Finished user conversion successfully!")
spark.stop()
