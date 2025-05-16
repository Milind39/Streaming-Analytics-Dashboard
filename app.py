import difflib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="StreamVerse Analytics",
    page_icon="🎬",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Streaming Platform Analysis Dashboard"
    }
)

# Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 2.5rem !important;
        color: #E50914 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .platform-card {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .recommendation-card {
        border-left: 4px solid #E50914;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        padding: 0.25rem 1rem;
    }
    .stButton button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #B20710;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Data loading with enhanced caching and error handling
@st.cache_data(show_spinner="Loading and processing data...")
def load_and_process_data():
    """Load and standardize data for all platforms with robust error handling"""
    platforms_data = {}
    platform_files = {
        'netflix': 'netflix_titles.csv',
        'amazon_prime': 'amazon_prime_titles.csv',
        'disney_plus': 'disney_plus_titles.csv',
        'hulu': 'hulu_titles.csv'
    }

    for platform, filename in platform_files.items():
        try:
            # Load data
            df = pd.read_csv(f"data/{filename}")

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')

            # Add platform identifier
            df['platform'] = platform.replace('_', ' ').title()

            # Handle dates
            if 'date_added' in df.columns:
                df['date_added'] = pd.to_datetime(
                    df['date_added'].astype(str).str.strip(),
                    errors='coerce'
                )
                df['year_added'] = df['date_added'].dt.year
                df['month_added'] = df['date_added'].dt.month_name()

            # Handle durations
            if 'duration' in df.columns:
                if 'type' in df.columns:
                    movies = df[df['type'] == 'Movie'].copy()
                    if 'duration' in movies.columns:
                        movies['duration'] = movies['duration'].str.extract('(\d+)').astype(float)
                        df.update(movies)

                    shows = df[df['type'] == 'TV Show'].copy()
                    if 'duration' in shows.columns:
                        shows['seasons'] = shows['duration'].str.extract('(\d+)').astype(float)
                        df.update(shows)

            # Fill missing values with safe defaults
            for col in ['director', 'cast', 'country', 'rating', 'listed_in', 'description']:
                if col in df.columns:
                    df[col].fillna('Unknown', inplace=True)

            platforms_data[platform] = df

        except Exception as e:
            st.error(f"Error loading {platform} data: {str(e)}")
            continue

    if not platforms_data:
        st.error("No data loaded. Please check your data files.")
        st.stop()

    # Combine all data
    all_data = pd.concat(platforms_data.values(), ignore_index=True)

    # Prepare features for recommendation system
    all_data['combined_features'] = (
            all_data['title'] + ' ' +
            all_data['director'] + ' ' +
            all_data['cast'] + ' ' +
            all_data['listed_in'] + ' ' +
            all_data['description']
    )

    # Create content age feature
    current_year = datetime.now().year
    all_data['content_age'] = current_year - all_data['release_year']

    return platforms_data, all_data


# Recommendation system with enhanced error handling
@st.cache_data(show_spinner="Building recommendation model...")
def get_recommendations(_all_data, title, num_rec=5):
    """Robust content-based recommendation engine"""
    try:
        if title not in _all_data['title'].values:
            similar_titles = difflib.get_close_matches(title, _all_data['title'].tolist(), n=1)
            if similar_titles:
                title = similar_titles[0]
            else:
                return pd.DataFrame()

        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
        tfidf_matrix = tfidf.fit_transform(_all_data['combined_features'])

        # Calculate cosine similarities
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get index of selected title
        idx = _all_data[_all_data['title'] == title].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top recommendations (skip the first which is the title itself)
        top_indices = [i[0] for i in sim_scores[1:num_rec + 1]]
        recommendations = _all_data.iloc[top_indices].copy()

        # Calculate similarity percentage
        recommendations['similarity'] = [round(score[1] * 100, 1) for score in sim_scores[1:num_rec + 1]]

        return recommendations

    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()


# Load data
platforms, all_data = load_and_process_data()

# Sidebar controls
st.sidebar.title("🎛️ Dashboard Controls")

selected_platforms = st.sidebar.multiselect(
    "Select Platforms",
    list(platforms.keys()),
    default=list(platforms.keys()),
    format_func=lambda x: x.replace('_', ' ').title()
)

analysis_type = st.sidebar.radio(
    "Select Analysis",
    [
        "📊 Platform Overview",
        "🎭 Genre Analysis",
        "📅 Release Trends",
        "⭐ Ratings Analysis",
        "⏱️ Duration Analysis",
        "🔍 Content Explorer",
        "🤖 Recommendation Engine"
    ]
)

# Filter data based on selection
filtered_data = all_data[all_data['platform'].isin(
    [p.replace('_', ' ').title() for p in selected_platforms]
)].copy()

# Main dashboard
st.markdown('<h1 class="title">StreamVerse Analytics</h1>', unsafe_allow_html=True)

if analysis_type == "📊 Platform Overview":
    st.header("Platform Comparison Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", filtered_data.shape[0])
    with col2:
        st.metric("Movies", filtered_data[filtered_data['type'] == 'Movie'].shape[0])
    with col3:
        st.metric("TV Shows", filtered_data[filtered_data['type'] == 'TV Show'].shape[0])
    with col4:
        st.metric("Avg Release Year", int(filtered_data['release_year'].mean()))

    # Platform distribution
    st.subheader("Content Distribution by Platform")
    platform_counts = filtered_data['platform'].value_counts().reset_index()
    fig = px.pie(
        platform_counts,
        names='platform',
        values='count',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

    # Content type by platform
    st.subheader("Content Type Distribution")
    fig = px.histogram(
        filtered_data,
        x='platform',
        color='type',
        barmode='group',
        color_discrete_sequence=['#E50914', '#221F1F'],
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "🎭 Genre Analysis":
    st.header("Genre Analysis")

    # Word Cloud
    st.subheader("Genre Word Cloud")
    all_genres = ' '.join(filtered_data['listed_in'].str.replace(', ', ' '))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='Reds',
        max_words=100
    ).generate(all_genres)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Top genres by platform
    st.subheader("Top Genres by Platform")
    platform_genres = filtered_data.groupby(['platform', 'listed_in']).size().reset_index(name='count')
    top_genres = platform_genres.sort_values(['platform', 'count'], ascending=False).groupby('platform').head(5)

    fig = px.bar(
        top_genres,
        x='count',
        y='listed_in',
        color='platform',
        orientation='h',
        facet_row='platform',
        height=800,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "📅 Release Trends":
    st.header("Content Release Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Release year distribution
        st.subheader("Content Release Years")
        fig = px.histogram(
            filtered_data,
            x='release_year',
            color='platform',
            nbins=50,
            marginal='box',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Added to platform over time
        if 'year_added' in filtered_data.columns:
            st.subheader("Content Added by Year")
            added_counts = filtered_data.groupby(['year_added', 'platform']).size().reset_index(name='count')
            fig = px.line(
                added_counts,
                x='year_added',
                y='count',
                color='platform',
                line_shape='spline',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

    # Age analysis
    st.subheader("Content Age Analysis")
    fig = px.box(
        filtered_data,
        x='platform',
        y='content_age',
        color='type',
        points='all',
        color_discrete_sequence=['#E50914', '#221F1F']
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "⭐ Ratings Analysis":
    st.header("Content Ratings Analysis")

    # Ratings distribution
    st.subheader("Ratings Distribution")
    ratings = filtered_data.groupby(['rating', 'platform']).size().reset_index(name='count')
    fig = px.bar(
        ratings,
        x='rating',
        y='count',
        color='platform',
        barmode='group',
        height=600,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

    # Ratings over time
    if 'year_added' in filtered_data.columns:
        st.subheader("Ratings Trend Over Time")
        ratings_time = filtered_data.groupby(['year_added', 'rating', 'platform']).size().reset_index(name='count')
        fig = px.line(
            ratings_time,
            x='year_added',
            y='count',
            color='rating',
            facet_row='platform',
            height=800,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "⏱️ Duration Analysis":
    st.header("Content Duration Analysis")

    # Prepare duration data
    movies = filtered_data[filtered_data['type'] == 'Movie'].copy()
    tv_shows = filtered_data[filtered_data['type'] == 'TV Show'].copy()

    col1, col2 = st.columns(2)

    with col1:
        # Movie durations
        st.subheader("Movie Durations (minutes)")
        if 'duration' in movies.columns:
            fig = px.box(
                movies,
                x='platform',
                y='duration',
                color='platform',
                points='all',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # TV Show seasons
        st.subheader("TV Show Seasons")
        if 'seasons' in tv_shows.columns:
            fig = px.histogram(
                tv_shows,
                x='seasons',
                color='platform',
                nbins=15,
                barmode='group',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

    # Duration vs Rating
    if 'duration' in movies.columns and 'rating' in movies.columns:
        st.subheader("Movie Duration vs Rating")
        fig = px.box(
            movies,
            x='rating',
            y='duration',
            color='platform',
            points='all',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "🔍 Content Explorer":
    st.header("Content Explorer")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        type_filter = st.selectbox("Content Type", ['All', 'Movie', 'TV Show'])
    with col2:
        year_filter = st.slider(
            "Release Year Range",
            min_value=int(filtered_data['release_year'].min()),
            max_value=int(filtered_data['release_year'].max()),
            value=(2010, 2023)
        )
    with col3:
        rating_filter = st.multiselect(
            "Content Rating",
            filtered_data['rating'].unique(),
            default=['TV-MA', 'TV-14', 'R', 'PG-13']
        )
    with col4:
        platform_filter = st.multiselect(
            "Platform",
            filtered_data['platform'].unique(),
            default=filtered_data['platform'].unique()
        )

    # Apply filters
    filtered_explorer = filtered_data.copy()
    if type_filter != 'All':
        filtered_explorer = filtered_explorer[filtered_explorer['type'] == type_filter]
    filtered_explorer = filtered_explorer[
        (filtered_explorer['release_year'] >= year_filter[0]) &
        (filtered_explorer['release_year'] <= year_filter[1])
        ]
    if rating_filter:
        filtered_explorer = filtered_explorer[filtered_explorer['rating'].isin(rating_filter)]
    if platform_filter:
        filtered_explorer = filtered_explorer[filtered_explorer['platform'].isin(platform_filter)]

    # Display results
    st.dataframe(
        filtered_explorer[[
            'title', 'type', 'platform', 'release_year', 'rating',
            'duration', 'seasons', 'listed_in', 'director'
        ]].sort_values('release_year', ascending=False),
        height=600,
        use_container_width=True,
        column_config={
            "title": "Title",
            "type": "Type",
            "platform": "Platform",
            "release_year": "Year",
            "rating": "Rating",
            "duration": "Duration (min)",
            "seasons": "Seasons",
            "listed_in": "Genres",
            "director": "Director"
        }
    )

elif analysis_type == "🤖 Recommendation Engine":
    st.header("🎬 Recommendation Engine")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Title selection with search
        search_term = st.text_input("Search for a title:", "")
        if search_term:
            matching_titles = [title for title in all_data['title'].unique()
                               if search_term.lower() in title.lower()]
        else:
            matching_titles = all_data['title'].unique()

        title_input = st.selectbox(
            "Select a title you enjoy:",
            sorted(matching_titles),
            index = min(100, len(matching_titles) - 1) if len(matching_titles) > 0 else 0
        )

        # Additional filters
        num_rec = st.slider("Number of recommendations:", 1, 10, 5)
        same_platform = st.checkbox("Only recommend from same platform", False)
        same_type = st.checkbox("Only recommend same content type", False)

    with col2:
        if title_input:
            try:
                # Display selected title info
                selected_title = all_data[all_data['title'] == title_input].iloc[0]

                st.markdown(f"""
                <div class="platform-card">
                    <h3>{selected_title['title']}</h3>
                    <p><strong>Platform:</strong> {selected_title.get('platform', 'N/A')}</p>
                    <p><strong>Type:</strong> {selected_title.get('type', 'N/A')}</p>
                    <p><strong>Release Year:</strong> {selected_title.get('release_year', 'N/A')}</p>
                    <p><strong>Rating:</strong> {selected_title.get('rating', 'N/A')}</p>
                    <p><strong>Genres:</strong> {selected_title.get('listed_in', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Get Recommendations", type="primary"):
                    with st.spinner('Analyzing similar content across platforms...'):
                        try:
                            # Get base recommendations
                            recommendations = get_recommendations(all_data, title_input, num_rec * 3)

                            if recommendations.empty:
                                st.warning("No recommendations found. Try a different title.")
                                st.stop()

                            # Apply filters if requested
                            if same_platform and 'platform' in recommendations.columns:
                                recommendations = recommendations[
                                    recommendations['platform'] == selected_title['platform']
                                    ]
                            if same_type and 'type' in recommendations.columns:
                                recommendations = recommendations[
                                    recommendations['type'] == selected_title['type']
                                    ]

                            # Limit to requested number
                            recommendations = recommendations.head(num_rec)

                            # Display recommendations
                            st.subheader("Recommended Titles")

                            for _, row in recommendations.iterrows():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{row.get('title', 'Unknown')} ({row.get('release_year', 'N/A')})</h4>
                                    <p>
                                        <strong>Platform:</strong> {row.get('platform', 'N/A')} | 
                                        <strong>Type:</strong> {row.get('type', 'N/A')} | 
                                        <strong>Rating:</strong> {row.get('rating', 'N/A')}<br>
                                        <strong>Genres:</strong> {row.get('listed_in', 'N/A')}<br>
                                        {f"<strong>Similarity:</strong> {row['similarity']}%" if 'similarity' in row else ''}
                                    </p>
                                    <p><em>{row.get('description', 'No description available')}</em></p>
                                </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
            except IndexError:
                st.warning("Selected title not found in dataset")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>StreamVerse Analytics Dashboard • Data from multiple sources • Updated: {date}</p>
</div>
""".format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)