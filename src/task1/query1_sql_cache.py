import time
from pyspark.sql import SparkSession

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("AdvDB_Query1_SQL_Cache") \
        .getOrCreate()

    # Reduce log verbosity
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read the Parquet dataset and register as Temp View
    reviews_df = spark.read.parquet("file:///home/nikolaos/ergasia_spark/parquet/yelp_review.parquet")
    reviews_df.createOrReplaceTempView("reviews")

    # 2. Apply initial filters and create the length_category column
    #    length() counts characters, not bytes
    filtered_df = spark.sql("""
        SELECT year, stars, useful, length_category
        FROM (
            SELECT year, stars, useful,
                CASE
                    WHEN length(text) < 250 THEN 'Σύντομη'
                    WHEN length(text) < 750 THEN 'Μεσαία'
                    WHEN length(text) < 2000 THEN 'Εκτενής'
                    WHEN length(text) >= 2000 THEN 'Πολύ εκτενής'
                END AS length_category
            FROM reviews
            WHERE year BETWEEN 2015 AND 2021
              AND useful >= 1
        )
        WHERE length_category IS NOT NULL
    """)
    filtered_df.createOrReplaceTempView("filtered_reviews")

    # Cache the filtered view (LAZY: stored in executor memory on the first action)
    spark.sql("CACHE LAZY TABLE filtered_reviews")

    # 3. Define the queries
    qa_df = spark.sql("""
        SELECT length_category,
               COUNT(*) AS review_count,
               AVG(stars) AS avg_stars,
               AVG(useful) AS avg_useful
        FROM filtered_reviews
        GROUP BY length_category
        ORDER BY avg_useful DESC
    """)

    qb_df = spark.sql("""
        SELECT year,
               length_category,
               cat_count,
               year_total,
               cat_count / year_total * 100 AS percentage
        FROM (
            SELECT year,
                   length_category,
                   COUNT(*) AS cat_count,
                   SUM(COUNT(*)) OVER (PARTITION BY year) AS year_total
            FROM filtered_reviews
            GROUP BY year, length_category
        )
        ORDER BY year,
                 CASE length_category
                     WHEN 'Σύντομη' THEN 1
                     WHEN 'Μεσαία' THEN 2
                     WHEN 'Εκτενής' THEN 3
                     ELSE 4
                 END
    """)

    # Physical plan (does not execute the query)
    qa_df.explain(mode="formatted")
    qb_df.explain(mode="formatted")

    # 4. Execute with cache: (a) and (b) one after the other
    #    collect() on the small final result both runs the full query and returns the rows
    start_a = time.time()
    rows_a = qa_df.collect()
    time_a = time.time() - start_a

    start_b = time.time()
    rows_b = qb_df.collect()
    time_b = time.time() - start_b

    print(f"RESULT cache time_a={time_a:.2f} time_b={time_b:.2f} total={time_a + time_b:.2f}")

    # Print the results (outside the timed section)
    print("\nQuery 1(a):")
    for r in rows_a:
        print(r)

    print("\nQuery 1(b):")
    for r in rows_b:
        print(r)

    spark.sql("UNCACHE TABLE filtered_reviews")
    spark.stop()

if __name__ == "__main__":
    main()