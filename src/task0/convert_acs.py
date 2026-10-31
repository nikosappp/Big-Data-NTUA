import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Convert_ACS_To_Parquet") \
    .getOrCreate()

raw_dir = os.path.expanduser("~/advdb/raw")
out_dir = os.path.expanduser("~/advdb/parquet")

# 2. Define explicit schema (zcta strictly as StringType)
acs_schema = StructType([
    StructField("zcta", StringType(), True),
    StructField("median_household_income", DoubleType(), True),
    StructField("population", LongType(), True)
])

print("--> Reading acs_zcta.csv with delimiter ';' and explicit schema...")
df_acs = spark.read \
    .option("header", "true") \
    .option("delimiter", ";") \
    .schema(acs_schema) \
    .csv(f"{raw_dir}/acs_zcta.csv")

print("--> Writing ACS to Parquet...")
df_acs.write.mode("overwrite").parquet(f"{out_dir}/acs_zcta.parquet")

print("--> Finished ACS conversion successfully!")
spark.stop()
