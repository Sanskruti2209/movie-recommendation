import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load and cache data
@st.cache_data
def load_data():
    md = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\movies_metadata.csv")
    links = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\links_small.csv")
    credits = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\credits.csv")
    keywords = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\keywords.csv")

    # Process ID column and filter for numeric IDs
    md = md[md['id'].str.isnumeric()]
    md['id'] = md['id'].astype(int)

    # Process genres
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # Extract year from release_date and handle missing dates
    md['release_date'] = pd.to_datetime(md['release_date'], errors='coerce')
    md['year'] = md['release_date'].dt.year

    # Filter for movies in links
    links = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
    smd = md[md['id'].isin(links)]

    # Combine overview and tagline for content-based filtering
    smd['tagline'] = smd['tagline'].fillna('')
    smd['overview'] = smd['overview'].fillna('')
    smd['description'] = smd['overview'] + " " + smd['tagline']

    # Merge with credits and keywords
    credits['id'] = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)
    smd = smd.merge(credits, on='id', how='left')
    smd = smd.merge(keywords, on='id', how='left')

    return smd.reset_index(), links

smd, links = load_data()
if smd is None:
    st.stop()

# TF-IDF Vectorizer and Cosine Similarity
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping indices and titles
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Train SVD Model
@st.cache_data
def train_svd():
    ratings = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\ratings_small.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    svd = SVD()
    svd.fit(trainset)
    return svd

svd = train_svd()

# Movie ID mapping
id_map = pd.read_csv(r"C:\Users\ACER\Downloads\Data\Data\links_small.csv")[['movieId', 'tmdbId']].dropna()
id_map['tmdbId'] = id_map['tmdbId'].astype(int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')

# Hybrid Recommendation Function
def hybrid(userId, title):
    if title not in indices:
        return ["Movie title not found in the dataset."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]  # Top 25

    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]

    # Estimate ratings using SVD
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    
    # Sort first by estimated rating, then by actual vote average (descending order)
    movies = movies.sort_values(by=['est', 'vote_average'], ascending=[False, False])
    
    return movies[['title', 'year', 'vote_count', 'vote_average', 'est']].head(10)


# Sidebar for page selection with a dropdown menu
page = st.sidebar.selectbox("Navigation", ["Home", "Charts", "Top Movies", "Random Movie", "Compare Movies"])

# Calculate genre counts globally so it's available in both "Charts" and "Top Movies"
genre_counts = smd['genres'].explode().value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

# Home Page - Movie Recommendation System
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #2D46B9;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #444;'>Discover movies based on your taste!</h4>", unsafe_allow_html=True)

    # Sidebar guide for the Home page
    st.sidebar.header("User Guide 📘")
    st.sidebar.write(""" 
    1. Enter a movie title you like.
    2. Enter your User ID for personalized suggestions.
    3. Click 'Recommend Movies' for suggestions.
    4. Use the 'Charts' and 'Top Movies' pages for additional insights.
    """)

    # User Input for movie recommendation
    col1, col2 = st.columns([2, 1])
    with col1:
        movie_name = st.text_input("🎞 Movie Title", placeholder="Enter a movie title")
    with col2:
        user_id = st.number_input("👤 User ID", min_value=1, step=1)

    # Display Recommendations
    if st.button("🎥 Recommend Movies"):
        if movie_name:
            recommendations = hybrid(user_id, movie_name)
            if isinstance(recommendations, list):
                st.error(recommendations[0])
            else:
                st.success("Top Recommended Movies for You:")
                # Beautify the recommendations table without highlighting
                st.dataframe(recommendations.style.set_table_attributes('style="width: 80%; margin: auto;"'))
        else:
            st.warning("Please enter a movie title to get recommendations.")

    # Search Feature
    search_term = st.text_input("🔍 Search Movies", placeholder="Search for a movie...")
    if st.button("Search"):
        if search_term:
            search_results = smd[smd['title'].str.contains(search_term, case=False)]
            if not search_results.empty:
                st.subheader("Search Results:")
                # Display the search results table without highlighting
                st.dataframe(search_results[['title', 'year', 'vote_average']].style.set_table_attributes('style="width: 80%; margin: auto;"').format({'vote_average': '{:.1f}'}))

                # Display detailed information for the first found movie
                if len(search_results) == 1:
                    movie = search_results.iloc[0]
                    st.markdown(f"### {movie['title']} ({movie['year']})")
                    st.markdown(f"*Vote Average:* {movie['vote_average']:.1f}")
                    st.markdown(f"*Genres:* {', '.join(movie['genres'])}")
                    st.markdown(f"*Overview:* {movie['overview']}")
            else:
                st.warning("No movies found matching your search.")
        else:
            st.warning("Please enter a movie title to search.")




# Charts Page
elif page == "Charts":
    st.markdown("<h1 style='text-align: center; color: #2D46B9;'>🔍 Data Insights</h1>", unsafe_allow_html=True)

    # Chart 1: Rating Distribution
    st.subheader("Distribution of Movie Ratings")
    rating_chart = alt.Chart(smd).mark_bar(color='teal').encode(
        alt.X("vote_average", bin=alt.Bin(maxbins=20), title="Rating"),
        alt.Y("count()", title="Number of Movies")
    )
    st.altair_chart(rating_chart, use_container_width=True)

    # Chart 2: Popular Genres
    st.subheader("Top Genres by Popularity")
    genre_chart = alt.Chart(genre_counts).mark_bar(color='orange').encode(
        x=alt.X("Genre", sort='-y', title="Genre"),
        y=alt.Y("Count", title="Number of Movies"),
    )
    st.altair_chart(genre_chart, use_container_width=True)

    # Chart 3: Top Rated Movies per Year
    st.subheader("Top Rated Movies per Year")
    top_rated_per_year = smd.sort_values(by=['year', 'vote_average'], ascending=[True, False]).drop_duplicates('year', keep='first')
    yearly_chart = alt.Chart(top_rated_per_year).mark_line(point=True).encode(
        x=alt.X("year", title="Year"),
        y=alt.Y("vote_average", title="Top Rating"),
        tooltip=["title", "year", "vote_average"]
    ).properties(width=600, height=400)
    st.altair_chart(yearly_chart, use_container_width=True)

# Top Movies Page
elif page == "Top Movies":
    st.markdown("<h1 style='text-align: center; color: #2D46B9;'>🌟 Top Movies</h1>", unsafe_allow_html=True)

    # Trending Movies
    trending_movies = smd.sort_values(by='vote_average', ascending=False).head(10)
    st.subheader("Top Movies:")
    # Display without highlighting
    st.dataframe(trending_movies[['title', 'year', 'vote_average']].style.set_table_attributes('style="width: 80%; margin: auto;"').format({'vote_average': '{:.1f}'}))

    # Filter Options
    st.sidebar.header("Filter Movies")
    genre_filter = st.sidebar.multiselect("Select Genres", options=smd['genres'].explode().unique().tolist())
    year_filter = st.sidebar.slider("Select Year Range", min_value=int(smd['year'].min()), max_value=int(smd['year'].max()), value=(2000, 2024))

    # Filter Movies Based on User Selection
    filtered_movies = smd[(smd['year'].between(year_filter[0], year_filter[1]))]
    if genre_filter:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre in x for genre in genre_filter))]

    # Sort by vote average in descending order
    filtered_movies = filtered_movies.sort_values(by='vote_average', ascending=False)

    # Display filtered movies
    st.subheader("Filtered Movies")
    # Display without highlighting
    st.dataframe(filtered_movies[['title', 'year', 'vote_average']].style.set_table_attributes('style="width: 80%; margin: auto;"').format({'vote_average': '{:.1f}'}))

# Random Movie Page
elif page == "Random Movie":
    st.markdown("<h1 style='text-align: center; color: #2D46B9;'>🎲 Random Movie Suggestion</h1>", unsafe_allow_html=True)
    
    # Genre selection for random movie with an "Any Genre" option
    genre_options = smd['genres'].explode().unique().tolist()
    genre_options.insert(0, "Any Genre")  # Add "Any Genre" option at the top
    selected_genre = st.selectbox("Select Genre", options=genre_options)
    
    if st.button("Get Random Movie"):
        if selected_genre == "Any Genre":
            random_movie = smd.sample()  # Get random movie from any genre
        else:
            random_movie = smd[smd['genres'].apply(lambda x: selected_genre in x)].sample()
        
        if not random_movie.empty:
            st.markdown(f"*Movie Title:* {random_movie['title'].values[0]} ({random_movie['year'].values[0]})")
            st.markdown(f"*Vote Average:* {random_movie['vote_average'].values[0]}")
            st.markdown(f"*Genres:* {', '.join(random_movie['genres'].values[0])}")
            st.markdown(f"*Overview:* {random_movie['overview'].values[0]}")
        else:
            st.warning("No movies found in this genre.")

# Comparison Page
elif page == "Compare Movies":
    st.markdown("<h1 style='text-align: center; color: #2D46B9;'>📊 Compare Movies</h1>", unsafe_allow_html=True)

    # User Input for Comparison
    compare_movies = st.multiselect("Select Movies to Compare", options=smd['title'].tolist(), max_selections=3)

    if len(compare_movies) == 2:
        # Fetch movie details for the selected movies, excluding overview
        movie_details = smd[smd['title'].isin(compare_movies)][['title', 'year', 'vote_average', 'vote_count', 'genres']]

        # Display the details in a table format
        st.subheader("Comparison Results:")
        if not movie_details.empty:
            # Adjust pandas display options for better visibility
            pd.set_option('display.max_colwidth', None)  # Allow unlimited width for columns
            
            # Display the DataFrame without the overview
            st.dataframe(movie_details.style.set_table_attributes('style="width: 80%; margin: auto;"').format({'vote_average': '{:.1f}'}))

            # Display movie overviews separately
            for index, row in smd[smd['title'].isin(compare_movies)].iterrows():
                st.markdown(f"### {row['title']} ({row['year']})")
                st.markdown(f"*Vote Average:* {row['vote_average']:.1f}  |  *Vote Count:* {row['vote_count']}  |  *Genres:* {', '.join(row['genres'])}")
                st.markdown(f"*Overview:* {row['overview']}\n\n")
        else:
            st.warning("No movie details found for the selected titles.")
    else:
        st.warning("Please select exactly two movies to compare.")