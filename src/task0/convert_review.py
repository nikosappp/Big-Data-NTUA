import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
from pyspark.sql.functions import col, year, to_timestamp

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_Review_To_Parquet_Partitioned") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema
review_schema = StructType([
    StructField("review_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("business_id", StringType(), True),
    StructField("stars", DoubleType(), True),
    StructField("useful", IntegerType(), True),
    StructField("funny", IntegerType(), True),
    StructField("cool", IntegerType(), True),
    StructField("text", StringType(), True),
    StructField("date", StringType(), True)
])

print("--> Reading review.json with explicit schema...")
df_review = spark.read.schema(review_schema).json(f"{raw_dir}/yelp_academic_dataset_review.json")

print("--> Extracting year column...")
df_review = df_review.withColumn("year", year(to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")))

print("--> Writing to Parquet format partitioned by year...")
df_review.write.mode("overwrite").partitionBy("year").parquet(f"{out_dir}/yelp_review.parquet")

print("--> Finished review conversion successfully!")
spark.stop()
