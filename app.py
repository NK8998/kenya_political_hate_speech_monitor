import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Artificial Intelligence-Driven Sentiment Analysis for Detecting Political Instability in Kenya", layout="wide")

st.title("🇰🇪 Kenya Political Sentiment Analysis Monitor Dashboard")
st.write("""
This tool analyzes political tweets, predicts hate-speech likelihood, and visualizes 
trends over time and across regions in Kenya.
""")

# -----------------------------------------
# MODEL LOADING
# -----------------------------------------

@st.cache_resource
def load_model_and_vectorizer():
    df = pd.read_csv("./content/kenya_hf.csv")
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].str.strip()

    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], df['labels'], test_size=0.2, random_state=42, stratify=df['labels']
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)

    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_tfidf, y_train)

    model = LinearSVC()
    model.fit(X_train_balanced, y_train_balanced)

    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()
st.success("Model loaded successfully!")

# -----------------------------------------
# FILE UPLOAD
# -----------------------------------------

uploaded = st.file_uploader("Upload a CSV file of tweets", type=["csv"])
st.write("### 📄 Required CSV Format")

st.info("""
Your CSV file **must contain** the following columns:

- **`text`** — *(required)*  
  Used for hate-speech prediction.

Optional columns for visualizations:

- **`date`** — *(optional but required for time-series trend)*  
  Must be parseable to a valid date.

- **`user_location`** — *(optional but required for county heatmap)*  
  Free-text is okay; the system will try to map it to Kenyan counties.

If these optional columns are missing, the app will still run but **maps and trend plots will be disabled**.
""")

if uploaded:
    df2 = pd.read_csv(uploaded)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(df2.head())

    if "text" not in df2.columns:
        st.error("Your CSV must contain a 'text' column.")
        st.stop()

    # Predict labels
    X2 = vectorizer.transform(df2['text'].astype(str))
    df2["predicted_label"] = model.predict(X2)

    # -----------------------------------------
    # SUMMARY CARDS
    # -----------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tweets Analyzed", len(df2))
    with col2:
        st.metric("Hate Speech Count", int(df2['predicted_label'].sum()))
    with col3:
        st.metric("Non-Hate Tweets", len(df2) - int(df2['predicted_label'].sum()))

    # -----------------------------------------
    # DATE PROCESSING
    # -----------------------------------------
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2["day"] = df2["date"].dt.date

        valid_dates = df2["day"].dropna()

        if valid_dates.empty:
            st.warning("No valid dates found in the dataset.")
        else:
            min_date = valid_dates.min()
            max_date = valid_dates.max()

            # DATE RANGE FILTER
            st.write("### 📅 Filter by Date Range")
            selected_range = st.slider(
                "Select date range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )
            df_filtered = df2[(df2["day"] >= selected_range[0]) & (df2["day"] <= selected_range[1])]

            # TREND PLOT
            st.write("### 📈 Hate-Speech Trend Over Time")
            trend = df_filtered.groupby("day")["predicted_label"].mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trend.index, trend.values)
            ax.set_title("Hate-Speech Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Average Hate Probability")
            ax.grid(True)
            st.pyplot(fig)

    # -----------------------------------------
    # LOCATION PROCESSING
    # -----------------------------------------
    st.write("### 🌍 Mapping User Locations")

    df2['clean_location'] = df2['user_location'].astype(str).str.lower()

    # Reduced for clarity; more mappings in the colab notebook
    county_keywords = {
        'nairobi': 'Nairobi',
        'kisumu': 'Kisumu',
        'mombasa': 'Mombasa',
        'nakuru': 'Nakuru',
        'eldoret': 'Uasin Gishu',
        'kericho': 'Kericho',
        'kajiado': 'Kajiado',
        'nyeri': 'Nyeri',
    }

    def map_location(loc):
        for k, v in county_keywords.items():
            if k in loc:
                return v
        return "Unknown"

    df2['county'] = df2['clean_location'].apply(map_location)

    # -----------------------------------------
    # KEYWORD EXPLORER
    # -----------------------------------------
    st.write("### 🔍 Keyword Explorer (Hate Speech Only)")
    hate_tweets = df2[df2['predicted_label'] == 1]['text']

    words = " ".join(hate_tweets).lower().split()
    word_freq = pd.Series(words).value_counts().head(20)

    fig3, ax3 = plt.subplots()
    word_freq.plot(kind="bar", figsize=(10,5), ax=ax3)
    ax3.set_title("Top Words in Hate-Speech Tweets")
    st.pyplot(fig3)

    # TEXT SEARCHER
    st.write("### 🔎 Search Tweets by Keyword")
    query = st.text_input("Enter keyword:")
    if query:
        results = df2[df2['text'].str.contains(query, case=False, na=False)]
        st.write(f"Results for '{query}':")
        st.dataframe(results[['text', 'predicted_label', 'county', 'date']].head(50))

    # -----------------------------------------
    # GEOGRAPHIC HOTSPOTS (Bar)
    # -----------------------------------------
    st.write("### 🔥 Geographic Hotspots by County")

    geo = df2.groupby("county")["predicted_label"].mean().sort_values()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    geo.plot(kind="barh", ax=ax2)
    ax2.set_title("Hate-Speech Hotspots by County")
    ax2.set_xlabel("Average Hate Speech Score")
    st.pyplot(fig2)

    # -----------------------------------------
    # MAP OF KENYA (Choropleth)
    # -----------------------------------------
    st.write("### 🗺️ Kenya Hate Speech Map")

    try:
        with open("./kenyan-counties.geojson", "r") as f:
            kenya_geo = json.load(f)

        df_map = df2.groupby("county")["predicted_label"].mean().reset_index()

        fig_map = px.choropleth(
            df_map,
            height=600,
            width=800,
            geojson=kenya_geo,
            locations="county",
            featureidkey="properties.COUNTY",
            color="predicted_label",
            color_continuous_scale="Reds",
            scope="africa",
            labels={'predicted_label': 'Hate Score'},
            title="Hate-Speech Intensity by County"
        )

        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, width="stretch")

    except FileNotFoundError:
        st.warning("GeoJSON file not found. Map disabled.")
    except Exception as e:
        st.error(f"Error loading map: {e}")

    # -----------------------------------------
    # DOWNLOAD OUTPUT
    # -----------------------------------------
    st.write("### 📥 Download Results")
    csv = df2.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV with Predictions", csv, "predictions.csv", "text/csv")
