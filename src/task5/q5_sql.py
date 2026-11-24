"""
Query 5 - SQL API

Usage:
    spark-submit ... q5_sql.py [mode]

    mode : run     -> print the plan and execute + time the query (default)
           explain -> print the plan only, do not execute
           stats   -> execute and also print the numbers needed for the report
                      (excluded postal codes / businesses, top-10 / bottom-10 lists).
                      The extra numbers are computed AFTER the timed section.

The executor configuration is given in spark-submit, e.g. 2 executors x (2 cores, 4 GB):
    --executor-cores 2 --executor-memory 4g --total-executor-cores 4
"""
import sys
import time
from pyspark.sql import SparkSession

PARQUET = "file:///home/nikolaos/ergasia_spark/parquet/"
VALID_MODES = {"run", "explain", "stats"}
METRICS = [("density", "(a) restaurants / 1000 residents"),
           ("avg_price", "(b) avg price range (1-4)"),
           ("avg_weekly_hours", "(c) avg weekly opening hours"),
           ("avg_review_stars", "(d) avg review stars 2017-2019")]
GROUPS = [("all", "all postal codes"),
          ("top10", "10 highest incomes"),
          ("bottom10", "10 lowest incomes")]


def main():
    # ------------------------------------------------------------------
    # 0. Command-line arguments
    # ------------------------------------------------------------------
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "run"
    if mode not in VALID_MODES:
        sys.exit(f"Usage: q5_sql.py [{'|'.join(sorted(VALID_MODES))}]")

    spark = SparkSession.builder.appName(f"AdvDB_Q5_SQL_{mode}").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    conf = spark.sparkContext.getConf()
    print(f"CONFIG mode={mode} "
          f"executor_cores={conf.get('spark.executor.cores', 'default')} "
          f"executor_memory={conf.get('spark.executor.memory', 'default')} "
          f"total_cores={conf.get('spark.cores.max', 'default')}")

    # ------------------------------------------------------------------
    # 1. Register the Parquet tables
    #    attributes and hours are already VARIANT in the Parquet file.
    #    review is partitioned by year.
    # ------------------------------------------------------------------
    spark.read.parquet(PARQUET + "yelp_business.parquet").createOrReplaceTempView("business")
    spark.read.parquet(PARQUET + "yelp_review.parquet").createOrReplaceTempView("reviews")
    spark.read.parquet(PARQUET + "acs_zcta.parquet").createOrReplaceTempView("acs")

    # ------------------------------------------------------------------
    # 2. Open restaurants with a 5-digit US postal code
    #    - exact category match on "Restaurants" (split on ", ")
    #    - price: STRING inside the variant ("1".."4", "None" or missing)
    #      -> try_variant_get + try_cast, so "None" becomes NULL (ANSI mode safe)
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT business_id,
               postal_code,
               CASE WHEN p BETWEEN 1 AND 4 THEN p END AS price,
               hours
        FROM (
            SELECT business_id, postal_code, hours,
                   try_cast(try_variant_get(attributes, '$.RestaurantsPriceRange2', 'string') AS INT) AS p
            FROM business
            WHERE is_open = 1
              AND array_contains(split(categories, ', '), 'Restaurants')
              AND postal_code RLIKE '^[0-9]{5}$'
        )
    """).createOrReplaceTempView("restaurants")

    # ------------------------------------------------------------------
    # 3. Weekly opening hours per restaurant
    #    hours = {"Monday": "11:0-22:30", ...}; one row per listed day via variant_explode.
    #    If closing <= opening, closing is on the next day (+24h), so "0:0-0:0" = 24 hours.
    #    Days that are not listed count as closed. Restaurants with NULL hours produce
    #    no rows here and are left out of the average (not counted as 0 hours).
    # ------------------------------------------------------------------
    spark.sql("""
        WITH days AS (
            SELECT r.business_id,
                   variant_get(h.value, '$', 'string') AS iv
            FROM restaurants r, LATERAL variant_explode(r.hours) AS h
        ),
        minutes AS (
            SELECT business_id,
                   try_cast(split(split(iv, '-')[0], ':')[0] AS INT) * 60
                 + try_cast(split(split(iv, '-')[0], ':')[1] AS INT)          AS open_min,
                   try_cast(split(split(iv, '-')[1], ':')[0] AS INT) * 60
                 + try_cast(split(split(iv, '-')[1], ':')[1] AS INT)          AS close_min
            FROM days
        )
        SELECT business_id,
               SUM(CASE WHEN close_min <= open_min THEN close_min - open_min + 1440
                        ELSE close_min - open_min END) / 60.0                 AS weekly_hours
        FROM minutes
        GROUP BY business_id
    """).createOrReplaceTempView("rest_hours")

    # ------------------------------------------------------------------
    # 4. Per postal code: restaurant count (>= 20), (b) avg price, (c) avg weekly hours
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT r.postal_code,
               COUNT(*)            AS n_restaurants,
               AVG(r.price)        AS avg_price,
               AVG(h.weekly_hours) AS avg_weekly_hours
        FROM restaurants r
        LEFT JOIN rest_hours h ON r.business_id = h.business_id
        GROUP BY r.postal_code
        HAVING COUNT(*) >= 20
    """).createOrReplaceTempView("zip_restaurants")

    # ------------------------------------------------------------------
    # 5. (d) avg review stars 2017-2019 per postal code
    #    The filter is on the partition column -> only year=2017..2019 are read.
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT r.postal_code,
               AVG(v.stars) AS avg_review_stars,
               COUNT(*)     AS n_reviews
        FROM reviews v
        JOIN restaurants r ON v.business_id = r.business_id
        WHERE v.year BETWEEN 2017 AND 2019
        GROUP BY r.postal_code
    """).createOrReplaceTempView("zip_reviews")

    # ------------------------------------------------------------------
    # 6. Valid ACS rows (Tip 6): drop negative special codes and population 0.
    #    Censored values (250001, 2499) are kept.
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT zcta, median_household_income AS income, population
        FROM acs
        WHERE median_household_income >= 0 AND population > 0
    """).createOrReplaceTempView("acs_valid")

    # ------------------------------------------------------------------
    # 7. One row per postal code with all four metrics + (a) density
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT z.postal_code                              AS zcta,
               a.income,
               a.population,
               z.n_restaurants,
               z.n_restaurants * 1000.0 / a.population    AS density,           -- (a)
               z.avg_price,                                                     -- (b)
               z.avg_weekly_hours,                                              -- (c)
               rv.avg_review_stars                                              -- (d)
        FROM zip_restaurants z
        JOIN acs_valid a        ON z.postal_code = a.zcta
        LEFT JOIN zip_reviews rv ON z.postal_code = rv.postal_code
    """).createOrReplaceTempView("zip_stats")

    # ------------------------------------------------------------------
    # 8. Final query: income correlation with (a)-(d), for all codes,
    #    the 10 highest and the 10 lowest incomes (ties: smaller zcta first).
    #    Single aggregation: corr() skips pairs with a NULL, so the CASE WHEN
    #    keeps only the rows of each group.
    # ------------------------------------------------------------------
    corr_cols = []
    for g, cond in [("all", "TRUE"), ("top10", "rank_desc <= 10"), ("bottom10", "rank_asc <= 10")]:
        corr_cols.append(f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) AS {g}_n")
        for m, _ in METRICS:
            corr_cols.append(
                f"corr(CASE WHEN {cond} THEN income END, CASE WHEN {cond} THEN {m} END) AS {g}_{m}")

    result = spark.sql(f"""
        WITH ranked AS (
            SELECT *,
                   row_number() OVER (ORDER BY income DESC, zcta ASC) AS rank_desc,
                   row_number() OVER (ORDER BY income ASC,  zcta ASC) AS rank_asc
            FROM zip_stats
        )
        SELECT {', '.join(corr_cols)}
        FROM ranked
    """)

    # ------------------------------------------------------------------
    # 9. Physical plan (does not execute the query)
    # ------------------------------------------------------------------
    result.explain(mode="formatted")

    if mode == "explain":
        print("RESULT explain_only (query not executed)")
        spark.stop()
        return

    # ------------------------------------------------------------------
    # 10. Execute and time: collect() on the single-row final result
    # ------------------------------------------------------------------
    start = time.perf_counter()
    row = result.collect()[0]
    elapsed = time.perf_counter() - start

    print(f"RESULT mode={mode} "
          f"executor_cores={conf.get('spark.executor.cores', 'default')} "
          f"executor_memory={conf.get('spark.executor.memory', 'default')} "
          f"time={elapsed:.2f}")

    # Print the correlations as a table (outside the timed section)
    print("\nQuery 5: Pearson correlation of median household income with each metric")
    print(f"{'group':<22}{'n':>4}" + "".join(f"{m:>20}" for m, _ in METRICS))
    for g, label in GROUPS:
        values = "".join(
            f"{row[f'{g}_{m}']:>20.4f}" if row[f'{g}_{m}'] is not None else f"{'NULL':>20}"
            for m, _ in METRICS)
        print(f"{label:<22}{row[f'{g}_n']:>4}{values}")

    # ------------------------------------------------------------------
    # 11. Extra numbers for the report (stats mode only, NOT timed)
    # ------------------------------------------------------------------
    if mode == "stats":
        def show(title, sql, n=30):
            print(f"\n=== {title} ===")
            spark.sql(sql).show(n, truncate=False)

        show("ACS rows dropped (whole file)", """
            SELECT COUNT(*)                                                          AS zctas_total,
                   SUM(CASE WHEN median_household_income < 0 THEN 1 ELSE 0 END)      AS special_code_income,
                   SUM(CASE WHEN population < 0 THEN 1 ELSE 0 END)                   AS special_code_population,
                   SUM(CASE WHEN median_household_income >= 0 AND population = 0
                            THEN 1 ELSE 0 END)                                       AS population_zero,
                   SUM(CASE WHEN median_household_income = 250001 THEN 1 ELSE 0 END) AS income_250001
            FROM acs
        """)

        show("Postal codes with >= 20 open restaurants, by ACS status", """
            SELECT CASE
                       WHEN a.zcta IS NULL                                      THEN 'not in ACS'
                       WHEN a.median_household_income < 0 OR a.population < 0   THEN 'ACS special code'
                       WHEN a.population = 0                                    THEN 'ACS population 0'
                       WHEN a.median_household_income = 250001                  THEN 'valid, income 250001'
                       ELSE 'valid'
                   END AS status,
                   COUNT(*) AS postal_codes,
                   SUM(z.n_restaurants) AS restaurants,
                   concat_ws(', ', slice(sort_array(collect_list(z.postal_code)), 1, 10)) AS examples
            FROM zip_restaurants z
            LEFT JOIN acs a ON z.postal_code = a.zcta
            GROUP BY 1 ORDER BY postal_codes DESC
        """)

        show("Open restaurants excluded because the postal code does not match a valid ZCTA", """
            WITH open_rest AS (
                SELECT business_id, postal_code
                FROM business
                WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
            )
            SELECT CASE
                       WHEN o.postal_code IS NULL OR NOT o.postal_code RLIKE '^[0-9]{5}$'
                                                                                THEN 'not a 5-digit code (Canada / empty)'
                       WHEN a.zcta IS NULL                                      THEN '5-digit, not in ACS'
                       WHEN a.median_household_income < 0 OR a.population <= 0  THEN '5-digit, invalid ACS row'
                       ELSE 'matched to a valid ZCTA'
                   END AS status,
                   COUNT(*) AS restaurants
            FROM open_rest o
            LEFT JOIN acs a ON o.postal_code = a.zcta
            GROUP BY 1 ORDER BY restaurants DESC
        """)

        show("Missing values inside the final postal codes", """
            SELECT COUNT(*)                                                    AS postal_codes,
                   SUM(CASE WHEN avg_price IS NULL THEN 1 ELSE 0 END)          AS no_price,
                   SUM(CASE WHEN avg_weekly_hours IS NULL THEN 1 ELSE 0 END)   AS no_hours,
                   SUM(CASE WHEN avg_review_stars IS NULL THEN 1 ELSE 0 END)   AS no_reviews_2017_2019
            FROM zip_stats
        """)

        cols = """zcta, income, population, n_restaurants,
                  ROUND(density, 3) AS density, ROUND(avg_price, 3) AS avg_price,
                  ROUND(avg_weekly_hours, 2) AS avg_weekly_hours,
                  ROUND(avg_review_stars, 3) AS avg_review_stars"""
        show("10 highest incomes", f"""
            SELECT {cols} FROM zip_stats ORDER BY income DESC, zcta ASC LIMIT 10
        """)
        show("10 lowest incomes", f"""
            SELECT {cols} FROM zip_stats ORDER BY income ASC, zcta ASC LIMIT 10
        """)

    spark.stop()


if __name__ == "__main__":
    main()