import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
from pyspark.sql.functions import col, year, to_timestamp, avg

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Benchmark_Review_Execution_Times") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

raw_dir = os.path.expanduser("~/advdb/raw")
parquet_dir = os.path.expanduser("~/advdb/parquet")
review_json_path = f"{raw_dir}/yelp_academic_dataset_review.json"
review_parquet_path = f"{parquet_dir}/yelp_review.parquet"

# Explicit schema definition for Review dataset
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

# -------------------------------------------------------------
# Scenario 1: JSON with Schema Inference
# -------------------------------------------------------------
print("\nStarting Scenario 1: JSON with Schema Inference")
start_time_s1 = time.time()

df_s1 = spark.read.json(review_json_path)
df_s1 = df_s1.withColumn("year", year(to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")))

query_s1 = df_s1.groupBy("year").agg(avg("stars").alias("avg_stars"))
query_s1.explain() 
res_s1 = query_s1.collect()

elapsed_s1 = time.time() - start_time_s1
print(f"Scenario 1 finished in: {elapsed_s1:.2f} seconds")

# -------------------------------------------------------------
# Scenario 2: JSON with Explicit Schema
# -------------------------------------------------------------
print("\nStarting Scenario 2: JSON with Explicit Schema")
start_time_s2 = time.time()

df_s2 = spark.read.schema(review_schema).json(review_json_path)
df_s2 = df_s2.withColumn("year", year(to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss")))

query_s2 = df_s2.groupBy("year").agg(avg("stars").alias("avg_stars"))
query_s2.explain() 
res_s2 = query_s2.collect()

elapsed_s2 = time.time() - start_time_s2
print(f"Scenario 2 finished in: {elapsed_s2:.2f} seconds")

# -------------------------------------------------------------
# Scenario 3: Parquet (Partitioned by year)
# -------------------------------------------------------------
print("\nStarting Scenario 3: Parquet (Partitioned)")
start_time_s3 = time.time()

df_s3 = spark.read.parquet(review_parquet_path) 

query_s3 = df_s3.groupBy("year").agg(avg("stars").alias("avg_stars"))
query_s3.explain() 
res_s3 = query_s3.collect()

elapsed_s3 = time.time() - start_time_s3
print(f"Scenario 3 finished in: {elapsed_s3:.2f} seconds")

# -------------------------------------------------------------
# Verify Output Equality
# -------------------------------------------------------------
print("\nVerifying Output Correctness")
sorted_s1 = sorted(res_s1, key=lambda x: str(x["year"]))
sorted_s2 = sorted(res_s2, key=lambda x: str(x["year"]))
sorted_s3 = sorted(res_s3, key=lambda x: str(x["year"]))

if sorted_s1 == sorted_s2 == sorted_s3:
    print("All 3 scenarios produced identical mathematical results!")
else:
    print("WARNING: Data mismatch detected between scenarios!")

print("\nAverage Stars per Year:")
for row in sorted(res_s3, key=lambda x: str(x["year"])):
    if row["year"] is not None:
        print(f"Year {row['year']}: {row['avg_stars']:.2f} stars")

spark.stop()
