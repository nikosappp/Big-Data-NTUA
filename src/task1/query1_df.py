import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, count, avg, sum
from pyspark.sql.window import Window

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("AdvDB_Query1_DataFrame") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read the Parquet dataset
    reviews_df = spark.read.parquet("../../parquet/yelp_review.parquet")

    # 2. Apply initial filters and create the length_category column
    # Filter for years 2015-2021 and useful >= 1
    # Use length(col("text")) to get character count, not bytes
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
        .otherwise("Πολύ εκτενής")
    ).select("year", "length_category", "stars", "useful")

    # 3. Define the queries as functions returning DataFrames
    def get_query_a(df):
        return df.groupBy("length_category").agg(
            count("*").alias("review_count"),
            avg("stars").alias("avg_stars"),
            avg("useful").alias("avg_useful")
        ).orderBy(col("avg_useful").desc())

    def get_query_b(df):
        # Calculate percentage per year using Window functions
        window_year = Window.partitionBy("year")
        
        return df.groupBy("year", "length_category") \
            .agg(count("*").alias("cat_count")) \
            .withColumn("year_total", sum("cat_count").over(window_year)) \
            .withColumn("percentage", (col("cat_count") / col("year_total")) * 100) \
            .orderBy("year", "length_category")

    # 4. Helper function to execute and measure time using 'noop' format
    def execute_and_measure(desc):
        print(f"\n--- Execution: {desc} ---")
        
        # Build logical plans
        qa_df = get_query_a(filtered_df)
        qb_df = get_query_b(filtered_df)

        # Εκτύπωση του πλάνου (αρκεί να εκτυπωθεί μόνο την πρώτη φορά, π.χ. στο "Without Cache")
        if desc == "Without Cache":
            print("\n--- Physical Plan for Query 1(a) ---")
            qa_df.explain("formatted")
        
        # Measure Query 1(a)
        start_a = time.time()
        qa_df.write.format("noop").mode("overwrite").save()
        end_a = time.time()
        
        # Measure Query 1(b)
        start_b = time.time()
        qb_df.write.format("noop").mode("overwrite").save()
        end_b = time.time()
        
        time_a = end_a - start_a
        time_b = end_b - start_b
        print(f"Query 1(a) Time: {time_a:.2f} seconds")
        print(f"Query 1(b) Time: {time_b:.2f} seconds")
        print(f"Total Time: {time_a + time_b:.2f} seconds")
       
    
    # 5. Execute without cache
    execute_and_measure("Without Cache")

    # 6. Execute with cache
    # Cache the filtered dataset
    filtered_df.cache()

    # First run with cache (materialization phase)
    print("\n[Note: First cached run includes materialization overhead]")
    execute_and_measure("With Cache (1st Run - Materialization)")

    # Second run with cache (reading purely from memory)
    execute_and_measure("With Cache (2nd Run - Reading from Memory)")

    # Cleanup
    filtered_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
