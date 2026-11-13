import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, count, avg, sum
from pyspark.sql.window import Window

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("AdvDB_Query1_DataFrame_Cache") \
        .getOrCreate()  

    # Reduce log verbosity
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read the Parquet dataset
    reviews_df = spark.read.parquet("file:///home/nikolaos/ergasia_spark/parquet/yelp_review.parquet")

    # 2. Apply initial filters and create the length_category column
    filtered_df = reviews_df.filter(
        (col("year") >= 2015) &
        (col("year") <= 2021) &
        (col("useful") >= 1)
    ).withColumn(
        "text_length", length(col("text"))
    ).withColumn(
        "length_category",
        when(col("text_length") < 250, "Σύντομη")
        .when(col("text_length") < 750, "Μεσαία")
        .when(col("text_length") < 2000, "Εκτενής")
        .when(col("text_length") >= 2000, "Πολύ εκτενής")
    ).filter(
        col("length_category").isNotNull()
    ).select("year", "stars", "useful", "length_category")

    # Cache the filtered dataset (lazy: stored in executor memory on the first action)
    filtered_df.cache()

    # 3. Define the queries as functions returning DataFrames
    def get_query_a(df):
        return df.groupBy("length_category").agg(
            count("*").alias("review_count"),
            avg("stars").alias("avg_stars"),
            avg("useful").alias("avg_useful")
        ).orderBy(col("avg_useful").desc())

    def get_query_b(df):
        window_year = Window.partitionBy("year")
        category_order = when(col("length_category") == "Σύντομη", 1) \
            .when(col("length_category") == "Μεσαία", 2) \
            .when(col("length_category") == "Εκτενής", 3) \
            .otherwise(4)
        return df.groupBy("year", "length_category") \
            .agg(count("*").alias("cat_count")) \
            .withColumn("year_total", sum("cat_count").over(window_year)) \
            .withColumn("percentage", (col("cat_count") / col("year_total")) * 100) \
            .orderBy(col("year"), category_order)

    qa_df = get_query_a(filtered_df)
    qb_df = get_query_b(filtered_df)

    # Physical plan (does not execute the query)
    qa_df.explain(mode="formatted")
    qb_df.explain(mode="formatted")

    # 4. Execute with cache: (a) and (b) one after the other
    start_a = time.time()
    rows_a = qa_df.collect()
    time_a1 = time.time() - start_a

    start_b = time.time()
    rows_b = qb_df.collect()
    time_b1 = time.time() - start_b

    print(f"RESULT cache_pass time_a={time_a1:.2f} time_b={time_b1:.2f} total={time_a1 + time_b1:.2f}")

    # Print the results (outside the timed section)
    print("\nQuery 1(a):")
    for r in rows_a:
        print(r)

    print("\nQuery 1(b):")
    for r in rows_b:
        print(r)

    filtered_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()