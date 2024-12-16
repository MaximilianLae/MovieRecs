from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, sum as Fsum
from pyspark.ml.recommendation import ALS


def initialize_spark():
    """
    Initialize and return a SparkSession.
    """
    return SparkSession.builder \
        .appName("MovieLens-Hybrid-Recommendation") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()


def load_data(spark, dataset_path, genre_columns):
    """
    Load data and prepare ALS and item metadata.
    """
    import pandas as pd

    # Load datasets
    u_data = pd.read_csv(f"{dataset_path}/ml-100k/u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
    u_item = pd.read_csv(f"{dataset_path}/ml-100k/u.item", sep="|", header=None, encoding="latin-1", 
                         names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL", 
                                "unknown"] + genre_columns)

    # Convert to PySpark DataFrames
    als_data_spark = spark.createDataFrame(u_data)
    u_item_spark = spark.createDataFrame(u_item[["item_id", "title"] + genre_columns])
    return als_data_spark, u_item_spark


def train_als_model(als_data_spark):
    """
    Train and return an ALS model.
    """
    train, _ = als_data_spark.randomSplit([0.8, 0.2], seed=42)
    als_model = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        nonnegative=True,
        coldStartStrategy="drop"
    ).fit(train)
    return als_model


def compute_user_preferences(als_data_spark, u_item_spark, genre_columns):
    """
    Compute user genre preferences for content-based filtering.
    """
    return als_data_spark.join(u_item_spark, on="item_id", how="left").groupBy("user_id").agg(
        *[Fsum(col(genre)).alias(genre) for genre in genre_columns]
    )


def identify_cold_start_items(als_data_spark, als_model, u_item_spark):
    """
    Identify cold-start items and return them with metadata.
    """
    all_items = als_data_spark.select("item_id").distinct()
    items_with_recommendations = als_model.recommendForAllItems(5).select("item_id").distinct()
    cold_start_items = all_items.subtract(items_with_recommendations)
    return cold_start_items.join(u_item_spark, on="item_id", how="left")


def hybrid_recommendation(user_id, als_model, als_data_spark, user_genre_preferences, 
                          cold_start_items_with_metadata, u_item_spark, genre_columns):
    """
    Generate hybrid recommendations for a user.
    Combines ALS, content-based filtering, and popularity fallback.
    """
    def recommend_als():
        user_recommendations = als_model.recommendForAllUsers(5)
        user_recs = user_recommendations.filter(col("user_id") == user_id)
        if user_recs.count() > 0:
            user_recs = user_recs.withColumn("rec", explode("recommendations")) \
                                 .select("user_id", col("rec.item_id").alias("item_id"), col("rec.rating").alias("predicted_rating"))
            return user_recs.join(u_item_spark, on="item_id", how="left")
        return None

    def recommend_cold_start():
        user_prefs = user_genre_preferences.filter(col("user_id") == user_id).collect()[0].asDict()
        scored_items = cold_start_items_with_metadata.withColumn("score", lit(0))
        for genre in genre_columns:
            scored_items = scored_items.withColumn("score", col("score") + col(genre) * lit(user_prefs.get(genre, 0)))
        return scored_items.orderBy(col("score").desc()).limit(5)

    def recommend_popularity():
        popular_items = als_data_spark.groupBy("item_id") \
                                      .agg(F.count("user_id").alias("interaction_count")) \
                                      .orderBy(col("interaction_count").desc())
        return popular_items.join(cold_start_items_with_metadata, on="item_id", how="inner") \
                            .join(u_item_spark, on="item_id", how="left") \
                            .select("title", "interaction_count") \
                            .limit(5)

    # Hybrid logic
    als_recommendations = recommend_als()
    if als_recommendations:
        return als_recommendations

    content_based_recommendations = recommend_cold_start()
    if content_based_recommendations.count() > 0:
        return content_based_recommendations

    return recommend_popularity()
