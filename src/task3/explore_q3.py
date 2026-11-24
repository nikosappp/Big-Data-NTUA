from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Explore_Q3").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

P = "file:///home/nikolaos/ergasia_spark/parquet/"
users = spark.read.parquet(P + "yelp_user.parquet")

# 1. Τι στήλες και τύπους έχει το user
print("=== SCHEMA user ===")
users.printSchema()

# 2. Πώς μοιάζουν το yelping_since και το elite
print("=== Δείγμα yelping_since / elite ===")
users.select("user_id", "yelping_since", "elite").show(10, truncate=False)

# 3. Πόσοι χρήστες ανά έτος εγγραφής
print("=== Χρήστες ανά έτος εγγραφής ===")
users.groupBy(F.year("yelping_since").alias("reg_year")).count() \
     .orderBy("reg_year").show(30)

# 4. Elite: κενά, null, και η ανωμαλία "20,20"
print("=== Elite: γενικά πλήθη ===")
users.select(
    F.count("*").alias("total_users"),
    F.sum(F.when(F.col("elite").isNull(), 1).otherwise(0)).alias("elite_null"),
    F.sum(F.when(F.col("elite") == "", 1).otherwise(0)).alias("elite_empty"),
    F.sum(F.when(F.col("elite").contains("20,20"), 1).otherwise(0)).alias("contains_20_20"),
).show()

# 5. Δείγμα τιμών elite που περιέχουν "20,20"
print("=== Δείγμα elite με '20,20' ===")
users.filter(F.col("elite").contains("20,20")) \
     .select("elite").show(15, truncate=False)

# 6. Όλα τα διαφορετικά «κομμάτια» μετά το split με κόμμα
print("=== Κομμάτια του elite μετά το split ===")
users.filter(F.col("elite") != "") \
     .select(F.explode(F.split("elite", ",")).alias("token")) \
     .groupBy("token").count().orderBy("token").show(50)

spark.stop()
