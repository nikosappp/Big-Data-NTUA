"""
Query 3 - SQL API

Usage:
    spark-submit ... q3_sql.py [hint] [mode]

    hint : none | broadcast | merge | shuffle_hash | shuffle_replicate_nl   (default: none)
           The hint is applied ONLY to the review JOIN user join, as /*+ HINT(u) */.
    mode : run     -> print the plan and execute (default)
           explain -> print the plan only, do not execute
           sample  -> execute on a 0.1% sample of review and user (for shuffle_replicate_nl)

AQE is controlled from spark-submit, e.g.:
    --conf spark.sql.adaptive.enabled=false
"""
import sys
import time
from pyspark.sql import SparkSession

PARQUET = "file:///home/nikolaos/ergasia_spark/parquet/"
VALID_HINTS = {"none", "broadcast", "merge", "shuffle_hash", "shuffle_replicate_nl"}
VALID_MODES = {"run", "explain", "sample"}
SAMPLE_FRACTION = 0.001
SAMPLE_SEED = 42


def main():
    # ------------------------------------------------------------------
    # 0. Command-line arguments
    # ------------------------------------------------------------------
    hint = sys.argv[1].lower() if len(sys.argv) > 1 else "none"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "run"
    if hint not in VALID_HINTS or mode not in VALID_MODES:
        sys.exit(f"Usage: q3_sql.py [{'|'.join(sorted(VALID_HINTS))}] [{'|'.join(sorted(VALID_MODES))}]")

    spark = SparkSession.builder \
        .appName(f"AdvDB_Q3_SQL_{hint}_{mode}") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    aqe = spark.conf.get("spark.sql.adaptive.enabled")
    print(f"CONFIG hint={hint} mode={mode} aqe={aqe}")

    # ------------------------------------------------------------------
    # 1. Register the Parquet tables as temp views
    #    Only the referenced columns are read (column pruning by the optimizer).
    # ------------------------------------------------------------------
    reviews_df = spark.read.parquet(PARQUET + "yelp_review.parquet")    # year = partition column = review year
    users_raw_df = spark.read.parquet(PARQUET + "yelp_user.parquet")
    business_df = spark.read.parquet(PARQUET + "yelp_business.parquet")

    # Optional 0.1% sample (only for the shuffle_replicate_nl experiment)
    if mode == "sample":
        reviews_df = reviews_df.sample(fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)
        users_raw_df = users_raw_df.sample(fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)

    reviews_df.createOrReplaceTempView("reviews")
    users_raw_df.createOrReplaceTempView("users_raw")
    business_df.createOrReplaceTempView("business")

    # ------------------------------------------------------------------
    # 2. Prepare users: registration year + cleaned list of elite years
    #
    #    elite is a comma-separated string, e.g. "2018,2019,20,20,2021".
    #    Year 2020 appears ONLY as the pair "20,20" (checked in explore_q3.py:
    #    token "20" appears 79,858 times = 2 x 39,929 users, token "2020" never).
    #    Rule: keep the 4-digit tokens as years, and add 2020 if token "20" exists.
    #    An empty string gives split -> [""] -> no 4-digit tokens -> empty array.
    # ------------------------------------------------------------------
    spark.sql("""
        SELECT user_id,
               year(to_timestamp(yelping_since)) AS reg_year,
               CASE
                   WHEN array_contains(split(elite, ','), '20') THEN
                       array_union(
                           transform(filter(split(elite, ','), t -> length(t) = 4), t -> CAST(t AS INT)),
                           array(2020))
                   ELSE
                       transform(filter(split(elite, ','), t -> length(t) = 4), t -> CAST(t AS INT))
               END AS elite_years
        FROM users_raw
    """).createOrReplaceTempView("users")

    # ------------------------------------------------------------------
    # 3. The hint for the review JOIN user join (on the user side, alias u)
    #    For BROADCAST the user table is broadcast,
    #    for SHUFFLE_HASH the user table is the build side.
    # ------------------------------------------------------------------
    hint_sql = "" if hint == "none" else f"/*+ {hint.upper()}(u) */"

    # ------------------------------------------------------------------
    # 4. Full query
    #    per_year : (a)-(d) from review JOIN user (hinted) JOIN business
    #    no_rev   : (e) users with no review at all -> LEFT ANTI JOIN (no hint)
    #    Full outer join so that no year is lost if it exists on one side only.
    #    The outer SELECT wraps the join so that ORDER BY reg_year is unambiguous.
    # ------------------------------------------------------------------
    result = spark.sql(f"""
        WITH per_year AS (
            SELECT {hint_sql}
                   u.reg_year,
                   COUNT(*)                                                    AS review_count,      -- (a)
                   AVG(r.stars)                                                AS avg_stars,         -- (b)
                   AVG(r.stars - b.stars)                                      AS avg_strictness,    -- (c)
                   AVG(CASE WHEN array_contains(u.elite_years, r.year)
                            THEN 1 ELSE 0 END) * 100                           AS elite_review_pct   -- (d)
            FROM reviews r
            JOIN users u    ON r.user_id = u.user_id
            JOIN business b ON r.business_id = b.business_id
            GROUP BY u.reg_year
        ),
        no_rev AS (
            SELECT u.reg_year,
                   COUNT(*) AS users_without_reviews                                                 -- (e)
            FROM users u
            LEFT ANTI JOIN reviews r ON u.user_id = r.user_id
            GROUP BY u.reg_year
        )
        SELECT *
        FROM (
            SELECT COALESCE(p.reg_year, n.reg_year)       AS reg_year,
                   COALESCE(p.review_count, 0)            AS review_count,
                   ROUND(p.avg_stars, 3)                  AS avg_stars,
                   ROUND(p.avg_strictness, 3)             AS avg_strictness,
                   ROUND(p.elite_review_pct, 2)           AS elite_review_pct,
                   COALESCE(n.users_without_reviews, 0)   AS users_without_reviews
            FROM per_year p
            FULL OUTER JOIN no_rev n ON p.reg_year = n.reg_year
        )
        ORDER BY reg_year
    """)

    # ------------------------------------------------------------------
    # 5. Physical plan (does not execute the query)
    # ------------------------------------------------------------------
    result.explain(mode="formatted")

    if mode == "explain":
        print("RESULT explain_only (query not executed)")
        spark.stop()
        return

    # ------------------------------------------------------------------
    # 6. Execute and time: collect() on the small final result (~19 rows)
    # ------------------------------------------------------------------
    start = time.perf_counter()
    rows = result.collect()
    elapsed = time.perf_counter() - start

    print(f"RESULT hint={hint} mode={mode} aqe={aqe} time={elapsed:.2f}")

    # Print the results (outside the timed section)
    print("\nQuery 3:")
    for r in rows:
        print(r)

    spark.stop()


if __name__ == "__main__":
    main()