import os
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Verify_All_Parquet") \
    .getOrCreate()

p_dir = os.path.expanduser("~/advdb/parquet")

datasets = [
    "acs_zcta.parquet",
    "glove_100d.parquet",
    "yelp_business.parquet",
    "yelp_checkin.parquet",
    "yelp_review.parquet",
    "yelp_tip.parquet",
    "yelp_user.parquet"
]

for name in datasets:
    path = os.path.join(p_dir, name)
    print("\n" + "=" * 60)
    print(f"Dataset: {name}")
    print("=" * 60)
    df = spark.read.parquet(path)
    df.printSchema()
    print(f"Row count: {df.count()}")
    df.show(2, truncate=50)

spark.stop()
