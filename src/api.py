from fastapi import FastAPI, HTTPException
from recommendation_engine import (initialize_spark, load_data, train_als_model, compute_user_preferences, 
                                   identify_cold_start_items, hybrid_recommendation)

app = FastAPI()

# Initialize Spark and load data
spark = initialize_spark()
dataset_path = "/home/maximilian.laechelin/pwc/Recommender/MovieRecommender/Data"
genre_columns = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", 
                 "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                 "Sci-Fi", "Thriller", "War", "Western"]

als_data_spark, u_item_spark = load_data(spark, dataset_path, genre_columns)
als_model = train_als_model(als_data_spark)
user_genre_preferences = compute_user_preferences(als_data_spark, u_item_spark, genre_columns)
cold_start_items_with_metadata = identify_cold_start_items(als_data_spark, als_model, u_item_spark)


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    recommendations = hybrid_recommendation(
        user_id=user_id,
        als_model=als_model,
        als_data_spark=als_data_spark,
        user_genre_preferences=user_genre_preferences,
        cold_start_items_with_metadata=cold_start_items_with_metadata,
        u_item_spark=u_item_spark,
        genre_columns=genre_columns
    )
    if recommendations.count() == 0:
        raise HTTPException(status_code=404, detail="No recommendations found for the user.")
    return recommendations.toPandas().to_dict(orient="records")
