"""
Data exploration for Query 5 (not timed, not part of the measurements).
Run with few resources, e.g.:
    spark-submit --master "local[2]" --driver-memory 2g src/task5/explore_q5.py
"""
from pyspark.sql import SparkSession

PARQUET = "file:///home/nikolaos/ergasia_spark/parquet/"

spark = SparkSession.builder.appName("Explore_Q5").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

spark.read.parquet(PARQUET + "yelp_business.parquet").createOrReplaceTempView("business")
spark.read.parquet(PARQUET + "acs_zcta.parquet").createOrReplaceTempView("acs")


def section(title, sql, n=30):
    print(f"\n=== {title} ===")
    spark.sql(sql).show(n, truncate=False)


# ----------------------------------------------------------------------
# 1. Schemas: types of postal_code, attributes, hours, and the ACS columns
# ----------------------------------------------------------------------
print("=== SCHEMA business ===")
spark.table("business").printSchema()
print("=== SCHEMA acs ===")
spark.table("acs").printSchema()

# ----------------------------------------------------------------------
# 2. "Restaurants": substring match vs exact category match
# ----------------------------------------------------------------------
section("Open restaurants: LIKE vs exact category", """
    SELECT
        SUM(CASE WHEN is_open = 1 AND categories LIKE '%Restaurants%' THEN 1 ELSE 0 END)                    AS open_like,
        SUM(CASE WHEN is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants') THEN 1 ELSE 0 END) AS open_exact,
        SUM(CASE WHEN categories IS NULL THEN 1 ELSE 0 END)                                                    AS categories_null
    FROM business
""")

section("Categories that contain 'Restaurants' but are not exactly 'Restaurants'", """
    SELECT cat, COUNT(*) AS n
    FROM business LATERAL VIEW explode(split(categories, ', ')) t AS cat
    WHERE cat LIKE '%Restaurants%' AND cat <> 'Restaurants'
    GROUP BY cat ORDER BY n DESC
""")

# ----------------------------------------------------------------------
# 3. postal_code formats among open restaurants
# ----------------------------------------------------------------------
section("postal_code formats (open restaurants)", """
    SELECT CASE
               WHEN postal_code IS NULL                       THEN 'null'
               WHEN postal_code = ''                          THEN 'empty'
               WHEN postal_code RLIKE '^[0-9]{5}$'            THEN 'US 5-digit'
               WHEN postal_code RLIKE '^[0-9]{5}-[0-9]{4}$'   THEN 'US ZIP+4'
               ELSE 'other'
           END AS fmt,
           COUNT(*) AS n
    FROM business
    WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
    GROUP BY 1 ORDER BY n DESC
""")

section("Examples of 'other' postal codes", """
    SELECT postal_code, state, COUNT(*) AS n
    FROM business
    WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
      AND NOT postal_code RLIKE '^[0-9]{5}$'
    GROUP BY postal_code, state ORDER BY n DESC
""", 15)

# ----------------------------------------------------------------------
# 4. attributes.RestaurantsPriceRange2 (already VARIANT in the Parquet file)
# ----------------------------------------------------------------------
section("Type of RestaurantsPriceRange2 inside attributes (schema_of_variant_agg)", """
    SELECT schema_of_variant_agg(variant_get(attributes, '$.RestaurantsPriceRange2')) AS price_schema
    FROM business
    WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
""")

section("Raw values of RestaurantsPriceRange2 (open restaurants)", """
    SELECT try_variant_get(attributes, '$.RestaurantsPriceRange2', 'string') AS price_raw,
           COUNT(*) AS n
    FROM business
    WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
    GROUP BY 1 ORDER BY n DESC
""")

# ----------------------------------------------------------------------
# 5. hours (already VARIANT, exploded per day)
# ----------------------------------------------------------------------
section("hours: sample of exploded (day, interval)", """
    SELECT b.business_id, h.key AS day, variant_get(h.value, '$', 'string') AS interval
    FROM business b, LATERAL variant_explode(b.hours) AS h
    WHERE b.is_open = 1 AND array_contains(split(b.categories, ', '), 'Restaurants')
""", 10)

section("hours: overview (open restaurants)", """
    WITH d AS (
        SELECT b.business_id,
               variant_get(h.value, '$', 'string') AS iv
        FROM business b, LATERAL variant_explode(b.hours) AS h
        WHERE b.is_open = 1 AND array_contains(split(b.categories, ', '), 'Restaurants')
    ),
    p AS (
        SELECT iv,
               CAST(split(split(iv, '-')[0], ':')[0] AS INT) * 60 + CAST(split(split(iv, '-')[0], ':')[1] AS INT) AS open_min,
               CAST(split(split(iv, '-')[1], ':')[0] AS INT) * 60 + CAST(split(split(iv, '-')[1], ':')[1] AS INT) AS close_min
        FROM d
    )
    SELECT COUNT(*)                                                           AS day_intervals,
           SUM(CASE WHEN iv RLIKE '^[0-9]{1,2}:[0-9]{1,2}-[0-9]{1,2}:[0-9]{1,2}$' THEN 0 ELSE 1 END) AS bad_format,
           SUM(CASE WHEN iv = '0:0-0:0' THEN 1 ELSE 0 END)                    AS is_0_0_to_0_0,
           SUM(CASE WHEN close_min <= open_min THEN 1 ELSE 0 END)             AS closes_next_day
    FROM p
""")

section("hours: how many restaurants have no hours at all", """
    SELECT SUM(CASE WHEN hours IS NULL THEN 1 ELSE 0 END) AS hours_null,
           COUNT(*)                                       AS open_restaurants
    FROM business
    WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
""")

# ----------------------------------------------------------------------
# 6. ACS: special codes, zero population, censored incomes (Tip 6)
# ----------------------------------------------------------------------
section("ACS overview", """
    SELECT COUNT(*)                                                               AS zctas,
           SUM(CASE WHEN median_household_income < 0 THEN 1 ELSE 0 END)           AS income_negative,
           SUM(CASE WHEN population < 0 THEN 1 ELSE 0 END)                        AS population_negative,
           SUM(CASE WHEN population = 0 THEN 1 ELSE 0 END)                        AS population_zero,
           SUM(CASE WHEN median_household_income = 250001 THEN 1 ELSE 0 END)      AS income_250001,
           SUM(CASE WHEN median_household_income = 2499 THEN 1 ELSE 0 END)        AS income_2499,
           SUM(CASE WHEN length(zcta) <> 5 THEN 1 ELSE 0 END)                     AS zcta_not_5_chars,
           SUM(CASE WHEN zcta LIKE '0%' THEN 1 ELSE 0 END)                        AS zcta_leading_zero
    FROM acs
""")

section("ACS: which negative codes appear", """
    SELECT 'income' AS col, median_household_income AS code, COUNT(*) AS n FROM acs
    WHERE median_household_income < 0 GROUP BY median_household_income
    UNION ALL
    SELECT 'population', population, COUNT(*) FROM acs
    WHERE population < 0 GROUP BY population
    ORDER BY col, n DESC
""")

# ----------------------------------------------------------------------
# 7. Coverage: postal codes with >= 20 open restaurants vs ACS
# ----------------------------------------------------------------------
section("Postal codes with >= 20 open restaurants and their ACS status", """
    WITH r AS (
        SELECT postal_code, COUNT(*) AS n_rest
        FROM business
        WHERE is_open = 1 AND array_contains(split(categories, ', '), 'Restaurants')
        GROUP BY postal_code
        HAVING COUNT(*) >= 20
    )
    SELECT CASE
               WHEN a.zcta IS NULL                                            THEN 'not in ACS'
               WHEN a.median_household_income < 0 OR a.population < 0         THEN 'ACS special code'
               WHEN a.population = 0                                          THEN 'ACS population 0'
               WHEN a.median_household_income = 250001                        THEN 'valid, income 250001'
               ELSE 'valid'
           END AS status,
           COUNT(*)      AS postal_codes,
           SUM(n_rest)   AS restaurants
    FROM r LEFT JOIN acs a ON r.postal_code = a.zcta
    GROUP BY 1 ORDER BY postal_codes DESC
""")

spark.stop()