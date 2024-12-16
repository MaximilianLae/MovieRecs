import streamlit as st
import requests
import pandas as pd

# Set API URL
API_URL = "http://127.0.0.1:8000/recommendations/"

# App title
st.title("Movie Recommendation System")
st.subheader("Get personalized movie recommendations based on your preferences!")

# Input for User ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    try:
        # Call the API
        response = requests.get(f"{API_URL}{user_id}")
        
        if response.status_code == 200:
            # Parse recommendations
            recommendations = response.json()

            # Convert to a DataFrame
            df = pd.DataFrame(recommendations)

            # Display recommendations
            st.write("### Recommendations:")
            st.dataframe(
                df[["title", "predicted_rating"]],
                width=700,
                height=400,
            )

            # Genre visualization (optional)
            st.write("### Movie Genres:")
            genre_columns = [
                "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                "Sci-Fi", "Thriller", "War", "Western"
            ]
            genre_summary = df[genre_columns].sum()
            st.bar_chart(genre_summary)

        else:
            st.error(f"Error: {response.status_code} - {response.json()['detail']}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
