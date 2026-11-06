import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, MapType
)
from pyspark.sql.functions import expr

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_Business_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema
business_schema = StructType([
    StructField("business_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("postal_code", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("stars", DoubleType(), True),
    StructField("review_count", IntegerType(), True),
    StructField("is_open", IntegerType(), True),
    StructField("attributes", StringType(), True),
    StructField("categories", StringType(), True),
    StructField("hours", StringType(), True)
])

print("--> Reading business.json with explicit schema...")
df_business = spark.read.schema(business_schema).json(f"{raw_dir}/yelp_academic_dataset_business.json")

df_business = (
    df_business
    .withColumn("attributes", expr("parse_json(attributes)"))
    .withColumn("hours", expr("parse_json(hours)"))
)

print("--> Writing to Parquet format...")
df_business.write.mode("overwrite").parquet(f"{out_dir}/yelp_business.parquet")

print("--> Finished business conversion successfully!")
spark.stop()
