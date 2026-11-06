import time
from pyspark.sql import SparkSession

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("AdvDB_Query1_SQL") \
        .getOrCreate()

    # Reduce log verbosity
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read the Parquet dataset and register as Temp View
    reviews_df = spark.read.parquet("../../parquet/yelp_review.parquet")
    reviews_df.createOrReplaceTempView("reviews")

    # 2. Apply initial filters and create the length_category column via SQL
    # Filter for years 2015-2021 and useful >= 1[cite: 3]
    # Use length(text) for character count, not bytes[cite: 5]
    filter_sql = """
        SELECT year, stars, useful,
            CASE 
                WHEN length(text) < 250 THEN 'Σύντομη'
                WHEN length(text) < 750 THEN 'Μεσαία'
                WHEN length(text) < 2000 THEN 'Εκτενής'
                ELSE 'Πολύ εκτενής'
            END AS length_category
        FROM reviews
        WHERE year BETWEEN 2015 AND 2021
        AND useful >= 1
    """
    
    # Create a DataFrame from the filtered SQL and register it as a new Temp View
    filtered_df = spark.sql(filter_sql)
    filtered_df.createOrReplaceTempView("filtered_reviews")

    # 3. Define SQL queries for (a) and (b)
    
    # Query 1(a): Group by category, calculate counts and averages[cite: 3]
    query_a_sql = """
        SELECT length_category,
               COUNT(*) AS review_count,
               AVG(stars) AS avg_stars,
               AVG(useful) AS avg_useful
        FROM filtered_reviews
        GROUP BY length_category
        ORDER BY avg_useful DESC
    """

    # Query 1(b): Group by year and category to find percentages using Window Functions[cite: 3]
    query_b_sql = """
        SELECT year, 
               length_category,
               (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY year)) AS percentage
        FROM filtered_reviews
        GROUP BY year, length_category
        ORDER BY year ASC, length_category ASC
    """

    # Helper function to execute and measure time using 'noop' format[cite: 9]
    def execute_and_measure(desc):
        print(f"\n--- Execution: {desc} ---")
        
        qa_df = spark.sql(query_a_sql)
        qb_df = spark.sql(query_b_sql)

        if desc == "Without Cache":
            print("\n--- Physical Plan for Query 1(a) ---")
            qa_df.explain("formatted")
        
        # Measure Query 1(a)[cite: 9]
        start_a = time.time()
        qa_df.write.format("noop").mode("overwrite").save()
        end_a = time.time()
        
        # Measure Query 1(b)[cite: 9]
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
    # Cache the filtered temporary view[cite: 7]
    spark.catalog.cacheTable("filtered_reviews")

    # First run with cache (materialization phase)
    print("\n[Note: First cached run includes materialization overhead]")
    execute_and_measure("With Cache (1st Run - Materialization)")

    # Second run with cache (reading purely from memory)
    execute_and_measure("With Cache (2nd Run - Reading from Memory)")

    # Cleanup
    spark.catalog.uncacheTable("filtered_reviews")
    spark.stop()

if __name__ == "__main__":
    main()
