"""
Query 3 - DataFrame API

Usage:
    spark-submit ... q3_df.py [hint] [mode]

    hint : none | broadcast | merge | shuffle_hash | shuffle_replicate_nl   (default: none)
           The hint is applied ONLY to the review JOIN user join.
    mode : run     -> print the plan and execute (default)
           explain -> print the plan only, do not execute
           sample  -> execute on a 0.1% sample of review and user (for shuffle_replicate_nl)

AQE is controlled from spark-submit, e.g.:
    --conf spark.sql.adaptive.enabled=false
"""
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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
        sys.exit(f"Usage: q3_df.py [{'|'.join(sorted(VALID_HINTS))}] [{'|'.join(sorted(VALID_MODES))}]")

    spark = SparkSession.builder \
        .appName(f"AdvDB_Q3_DF_{hint}_{mode}") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    aqe = spark.conf.get("spark.sql.adaptive.enabled")
    print(f"CONFIG hint={hint} mode={mode} aqe={aqe}")

    # ------------------------------------------------------------------
    # 1. Read only the columns we need (column pruning)
    # ------------------------------------------------------------------
    reviews = spark.read.parquet(PARQUET + "yelp_review.parquet") \
        .select("user_id", "business_id", "stars", "year")          # year = partition column = review year

    users_raw = spark.read.parquet(PARQUET + "yelp_user.parquet") \
        .select("user_id", "yelping_since", "elite")

    business = spark.read.parquet(PARQUET + "yelp_business.parquet") \
        .select("business_id", F.col("stars").alias("business_stars"))

    # ------------------------------------------------------------------
    # 2. Prepare users: registration year + cleaned list of elite years
    #
    #    elite is a comma-separated string, e.g. "2018,2019,20,20,2021".
    #    Year 2020 appears ONLY as the pair "20,20" (checked in explore_q3.py:
    #    token "20" appears 79,858 times = 2 x 39,929 users, token "2020" never).
    #    Rule: keep the 4-digit tokens as years, and add 2020 if token "20" exists.
    #    An empty string gives split -> [""] -> no 4-digit tokens -> empty array.
    # ------------------------------------------------------------------
    tokens = F.split(F.col("elite"), ",")
    four_digit_years = F.transform(
        F.filter(tokens, lambda t: F.length(t) == 4),
        lambda t: t.cast("int")
    )
    elite_years = F.when(
        F.array_contains(tokens, "20"),
        F.array_union(four_digit_years, F.array(F.lit(2020)))
    ).otherwise(four_digit_years)

    users = users_raw.select(
        "user_id",
        F.year(F.to_timestamp("yelping_since")).alias("reg_year"),
        elite_years.alias("elite_years")
    )

    # Optional 0.1% sample (only for the shuffle_replicate_nl experiment)
    if mode == "sample":
        reviews = reviews.sample(fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)
        users = users.sample(fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)

    # ------------------------------------------------------------------
    # 3. (a)-(d): review JOIN user (with the requested hint) JOIN business
    #    The hint goes on the user side, so for BROADCAST the user table is broadcast
    #    and for SHUFFLE_HASH the user table is the build side.
    # ------------------------------------------------------------------
    users_for_join = users if hint == "none" else users.hint(hint)

    reviews_full = reviews \
        .join(users_for_join, on="user_id", how="inner") \
        .join(business, on="business_id", how="inner")

    per_year = reviews_full.groupBy("reg_year").agg(
        F.count("*").alias("review_count"),                                      # (a)
        F.avg("stars").alias("avg_stars"),                                       # (b)
        F.avg(F.col("stars") - F.col("business_stars")).alias("avg_strictness"), # (c)
        (F.avg(F.when(F.array_contains("elite_years", F.col("year")), 1)
                .otherwise(0)) * 100).alias("elite_review_pct")                  # (d)
    )

    # ------------------------------------------------------------------
    # 4. (e): users with no review at all -> anti-join (no hint here)
    # ------------------------------------------------------------------
    users_without_reviews = users \
        .join(reviews.select("user_id"), on="user_id", how="left_anti") \
        .groupBy("reg_year") \
        .agg(F.count("*").alias("users_without_reviews"))

    # ------------------------------------------------------------------
    # 5. Combine (a)-(d) with (e) per registration year, sort ascending
    #    Full outer join so that no year is lost if it exists on one side only.
    # ------------------------------------------------------------------
    result = per_year \
        .join(users_without_reviews, on="reg_year", how="full_outer") \
        .select(
            "reg_year",
            F.coalesce("review_count", F.lit(0)).alias("review_count"),
            F.round("avg_stars", 3).alias("avg_stars"),
            F.round("avg_strictness", 3).alias("avg_strictness"),
            F.round("elite_review_pct", 2).alias("elite_review_pct"),
            F.coalesce("users_without_reviews", F.lit(0)).alias("users_without_reviews"),
        ) \
        .orderBy("reg_year")

    # ------------------------------------------------------------------
    # 6. Physical plan (does not execute the query)
    # ------------------------------------------------------------------
    result.explain(mode="formatted")

    if mode == "explain":
        print("RESULT explain_only (query not executed)")
        spark.stop()
        return

    # ------------------------------------------------------------------
    # 7. Execute and time: collect() on the small final result (~19 rows)
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