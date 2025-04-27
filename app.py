# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import time
import numpy as np
from serpapi import GoogleSearch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel
from googletrans import Translator
import spacy
import plotly.express as px
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data at startup
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="VibeChek AI Dashboard", layout="wide")

# Define standard figure sizes for consistent display
FIGURE_SIZES = {
    "large": (7, 3.5),      # For main visualizations
    "medium": (4, 2.5),     # For secondary visualizations 
    "small": (3.5, 2.3),    # For compact visualizations
    "pie": (3, 2.5)         # Specifically for pie charts
}

# Add custom CSS for better spacing and containment
st.markdown("""
<style>
    .plot-container {
        max-width: 90%;
        margin: 0 auto;
    }
    .section-divider {
        margin-top: 2em;
        margin-bottom: 1em;
    }
    .subsection-divider {
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    /* Make all charts and visualizations more compact */
    .stPlotlyChart, .stChart {
        max-width: 90% !important;
        margin: 0 auto !important;
    }
    /* Add custom scaling for better fit */
    div[data-testid="stImage"] img {
        max-width: 90% !important;
        display: block !important;
        margin: 0 auto !important;
    }
    /* Style for semantic insights */
    .insight-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4e8df5;
    }
    .topic-card {
        background-color: #f0f7ff;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #3366cc;
    }
    .phrase-header {
        font-weight: bold;
        color: #333;
    }
    .phrase-count {
        color: #666;
        font-size: 0.9em;
    }
    .phrase-sentiment {
        font-style: italic;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Helper functions
def clean_text(text):
    """Clean text by removing URLs, special characters, and converting to lowercase"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.strip().lower()

def vader_sentiment(text):
    """Basic VADER sentiment analysis"""
    if not text:
        return "Neutral"
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

def enhanced_business_sentiment(text):
    """Enhanced business-specific sentiment analysis"""
    if not text:
        return "Neutral"
    
    # Get the base VADER scores
    score = sia.polarity_scores(text)["compound"]
    
    # Business-specific sentiment boosters
    business_positive = [
        'recommend', 'excellent', 'amazing', 'love', 'best', 
        'friendly', 'helpful', 'clean', 'professional', 'fresh',
        'worth', 'perfect', 'fantastic', 'awesome', 'definitely'
    ]
    
    business_negative = [
        'waste', 'overpriced', 'rude', 'slow', 'dirty',
        'terrible', 'horrible', 'avoid', 'disappointing', 'cold',
        'manager', 'complained', 'waiting', 'problem', 'never again'
    ]
    
    # Check for business-specific terms and adjust score
    text_lower = text.lower()
    
    # Apply modest boosts to the compound score for business-specific terms
    for term in business_positive:
        if term in text_lower:
            score = min(1.0, score + 0.05)
            
    for term in business_negative:
        if term in text_lower:
            score = max(-1.0, score - 0.05)
    
    # Adjust thresholds for business reviews (they tend to be more polarized)
    if score >= 0.1:
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_top_words(reviews, n=10):
    """Extract and count the most common words in reviews"""
    if not hasattr(reviews, 'any') or not reviews.any():
        return []
            
    # Combine all review text
    all_text = " ".join(reviews)
    
    # Split into words and count
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Simple stopwords filtering
    stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'was', 'for', 
                 'of', 'with', 'on', 'at', 'by', 'this', 'that', 'but', 'are', 
                 'be', 'or', 'have', 'has', 'had', 'not', 'what', 'all', 'were', 
                 'when', 'where', 'who', 'which', 'their', 'they', 'them', 'there',
                 'from', 'out', 'some', 'would', 'about', 'been', 'many', 'us', 'we'}
    
    # Filter out stopwords and short words
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Return top N words
    return word_counts.most_common(n)

# NEW FUNCTIONS FOR SEMANTIC ANALYSIS

def translate_text(text, target_language='en'):
    """Translate text to target language (default is English)"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def detect_language(text):
    """Detect the language of the text"""
    if not text or not isinstance(text, str):
        return "en"  # Default to English
    
    try:
        translator = Translator()
        detected = translator.detect(text)
        return detected.lang
    except:
        return "en"  # Default to English if detection fails

def extract_ngrams(text, min_n=2, max_n=4, top_n=15):
    """Extract most common n-grams (phrases) from text"""
    if not text or not isinstance(text, str):
        return []
    
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
    
    # Get n-grams from min_n to max_n
    all_ngrams = []
    for n in range(min_n, max_n + 1):
        n_grams = list(ngrams(filtered_tokens, n))
        all_ngrams.extend([' '.join(g) for g in n_grams])
    
    # Count and return top n-grams
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_n)

def perform_topic_modeling(texts, num_topics=5, num_words=8):
    """Perform LDA topic modeling on reviews"""
    if not texts or len(texts) < 5:
        return None, None
    
    # Preprocess texts
    tokenized_texts = []
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        
        # Tokenize and filter
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
        
        if filtered_tokens:
            tokenized_texts.append(filtered_tokens)
    
    if not tokenized_texts:
        return None, None
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Build LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, dictionary

def get_sentiment_for_phrase(phrase, reviews_df):
    """Calculate sentiment score for a specific phrase"""
    # Find reviews containing the phrase
    matching_reviews = reviews_df[reviews_df['Cleaned_Review'].str.contains(phrase, na=False)]
    
    if len(matching_reviews) == 0:
        return 0  # Neutral if no matches
    
    # Get average sentiment
    sentiment_counts = matching_reviews['Sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    # Calculate sentiment ratio (-1 to +1)
    total = positive_count + negative_count + neutral_count
    if total == 0:
        return 0
    
    sentiment_score = (positive_count - negative_count) / total
    return sentiment_score

def extract_common_contexts(phrase, reviews, max_examples=3):
    """Extract common contexts where a phrase appears"""
    contexts = []
    
    for review in reviews:
        if not isinstance(review, str) or not review:
            continue
            
        if phrase.lower() in review.lower():
            sentences = sent_tokenize(review)
            for sentence in sentences:
                if phrase.lower() in sentence.lower():
                    contexts.append(sentence)
                    if len(contexts) >= max_examples:
                        return contexts
    
    return contexts

def get_phrase_frequency_data(reviews_df, sentiment_filter=None):
    """Get phrase frequency data with sentiment breakdown"""
    if sentiment_filter:
        filtered_df = reviews_df[reviews_df['Sentiment'] == sentiment_filter]
    else:
        filtered_df = reviews_df
    
    if len(filtered_df) == 0:
        return []
    
    all_text = " ".join(filtered_df['Cleaned_Review'].dropna())
    
    # Extract 2-4 word phrases
    phrases = extract_ngrams(all_text, min_n=2, max_n=4, top_n=20)
    
    # Add sentiment information for each phrase
    phrase_data = []
    for phrase, count in phrases:
        sentiment_score = get_sentiment_for_phrase(phrase, reviews_df)
        sentiment_label = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
        
        # Get example contexts
        contexts = extract_common_contexts(phrase, reviews_df['Cleaned_Review'].dropna(), max_examples=2)
        
        phrase_data.append({
            'phrase': phrase,
            'count': count,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'contexts': contexts
        })
    
    return phrase_data

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# App title and description
st.title("🧠 VibeChek: Google Review Analyzer")

st.markdown("""
**Don't know your Place ID?**
🔗 [Find your Google Place ID here](https://developers.google.com/maps/documentation/places/web-service/place-id)
Search for your business and copy the Place ID.
""")

# Get API Key from secrets
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
except Exception:
    SERPAPI_KEY = st.text_input("Enter your SerpAPI Key", type="password")
    if not SERPAPI_KEY:
        st.warning("Please enter your SerpAPI Key to continue")
        st.stop()

# Initialize session state for storing data
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = None

# Input Place ID
place_id = st.text_input("📍 Enter Google Place ID")

# Language options
languages = ["auto", "en", "sw", "fr", "es", "de", "zh-cn", "ar", "ru", "pt", "ja"]
source_language = st.selectbox("Source Language", languages, index=0)

# Max Reviews
max_reviews = st.slider("🔄 How many reviews to fetch?", min_value=50, max_value=500, step=50, value=150)

if st.button("🚀 Fetch & Analyze Reviews") and place_id:
    try:
        with st.spinner("Fetching reviews from Google Maps..."):
            # Create params with error handling
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY,
            }
            
            # Test API connection with just one request first
            try:
                # Explicitly use the correct import path
                test_search = GoogleSearch(params)
                test_results = test_search.get_dict()
                
                if "error" in test_results:
                    st.error(f"API Error: {test_results['error']}")
                    st.stop()
                    
                if "reviews" not in test_results:
                    st.warning("No reviews found for this Place ID. Please verify the ID is correct.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error connecting to SerpAPI: {str(e)}")
                st.stop()
            
            # Now fetch all reviews
            all_reviews = []
            start = 0
            
            progress_bar = st.progress(0)
            
            # Make multiple requests with pagination
            while len(all_reviews) < max_reviews:
                params["start"] = start
                
                try:
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    reviews = results.get("reviews", [])
                    
                    if not reviews:
                        break
                        
                    all_reviews.extend(reviews)
                    start += len(reviews)
                    
                    # Update progress
                    progress = min(len(all_reviews) / max_reviews, 1.0)
                    progress_bar.progress(progress)
                    
                    # Sleep to respect API rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    st.warning(f"Error during pagination (fetched {len(all_reviews)} reviews so far): {str(e)}")
                    break
            
            if not all_reviews:
                st.error("No reviews could be fetched. Please check your Place ID and API key.")
                st.stop()
                
            df = pd.DataFrame(all_reviews[:max_reviews])
            
            # Handle missing columns - sometimes SerpAPI response structure varies
            for col in ['snippet', 'rating', 'time']:
                if col not in df.columns:
                    df[col] = None
            
            # Basic data validation
            df = df.dropna(subset=['snippet'])
            
            if len(df) == 0:
                st.error("No valid reviews found after filtering.")
                st.stop()
                
            st.session_state.reviews_df = df
            st.success(f"✅ {len(df)} reviews fetched!")
    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()
    
    # Process the data
    try:
        with st.spinner("Processing reviews..."):
            # Handle translation if needed
            if source_language != "en" and source_language != "auto":
                st.info("Translating reviews... this may take a moment")
                df['Translated_Review'] = df['snippet'].apply(lambda x: translate_text(x) if x else "")
                df["Cleaned_Review"] = df['Translated_Review'].apply(clean_text)
            elif source_language == "auto":
                # Auto-detect and translate non-English reviews
                df['Detected_Language'] = df['snippet'].apply(lambda x: detect_language(x) if x else "en")
                
                # Translate non-English reviews
                df['Translated_Review'] = df.apply(
                    lambda row: translate_text(row['snippet']) 
                    if row['Detected_Language'] != 'en' and row['snippet'] 
                    else row['snippet'], axis=1
                )
                
                df["Cleaned_Review"] = df['Translated_Review'].apply(clean_text)
            else:
                # Just clean English reviews
                df["Cleaned_Review"] = df["snippet"].apply(clean_text)
            
            # Apply enhanced sentiment analysis
            df["Sentiment"] = df["Cleaned_Review"].apply(enhanced_business_sentiment)
            
            # Simple ratings analysis with colorful bars
            if "rating" in df.columns and df["rating"].notna().any():
                # Create a figure with adjusted height to accommodate the legend
                fig, ax = plt.subplots(figsize=(FIGURE_SIZES["medium"][0], FIGURE_SIZES["medium"][1] + 0.5))
                
                # Ensure we're working with numeric ratings and convert to integers if needed
                df['rating_num'] = pd.to_numeric(df['rating'], errors='coerce')
                rating_counts = df['rating_num'].value_counts().sort_index()
                
                # Define color map - use both integer and float keys to handle both types
                color_map = {
                    1: '#d73027', 1.0: '#d73027',     # Red
                    2: '#fc8d59', 2.0: '#fc8d59',     # Orange
                    3: '#ffffbf', 3.0: '#ffffbf',     # Yellow
                    4: '#91cf60', 4.0: '#91cf60',     # Light green
                    5: '#1a9850', 5.0: '#1a9850',     # Dark green
                }
                
                # Explicitly create arrays for plotting rather than using pandas series directly
                ratings = rating_counts.index.tolist()
                counts = rating_counts.values.tolist()
                
                # Create a list of colors for each bar - with fallback default color
                bar_colors = []
                for rating in ratings:
                    if rating in color_map:
                        bar_colors.append(color_map[rating])
                    else:
                        # Fallback to blue if we somehow get an unexpected rating
                        bar_colors.append('#4575b4')
                
                # Create the plot
                bars = ax.bar(ratings, counts, color=bar_colors)
                
                # Add rating values on top of each bar - with smaller font
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            str(int(count)), ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Set labels and title with smaller font
                ax.set_xlabel("Rating", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title("Rating Distribution", fontsize=10)
                
                # Set x-axis ticks - explicitly use only integer ratings from 1-5
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
                
                # Add a legend explaining the color scheme - with smaller font
                # MOVED BELOW THE PLOT
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#d73027', label='1 Star'),
                    Patch(facecolor='#fc8d59', label='2 Stars'),
                    Patch(facecolor='#ffffbf', label='3 Stars'),
                    Patch(facecolor='#91cf60', label='4 Stars'),
                    Patch(facecolor='#1a9850', label='5 Stars')
                ]
                
                # Place legend below the plot
                ax.legend(handles=legend_elements, 
                          title="Rating Colors", 
                          loc='upper center', 
                          bbox_to_anchor=(0.5, -0.15),
                          ncol=5, 
                          fontsize=7, 
                          title_fontsize=8)
                
                # Adjust figure to make room for the legend
                plt.subplots_adjust(bottom=0.2)
                
                # Improve layout
                plt.tight_layout()
                
                # Show the plot with container
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Show sentiment distribution
            st.subheader("📊 Sentiment Analysis")
            sentiment_counts = df["Sentiment"].value_counts()
            
            # RESIZED pie chart and smaller fonts
            fig, ax = plt.subplots(figsize=FIGURE_SIZES["pie"])
            colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%',
                colors=[colors.get(x, 'blue') for x in sentiment_counts.index],
                textprops={'fontsize': 8}
            )
            for autotext in autotexts:
                autotext.set_fontsize(8)
            
            ax.set_title("Sentiment Distribution", fontsize=10)
            plt.tight_layout()
            
            # Show the plot with container
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show the data
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📋 Review Data")
            
            # Show translation data if applicable
            if 'Translated_Review' in df.columns:
                display_cols = ["snippet", "Translated_Review", "Sentiment"]
            else:
                display_cols = ["snippet", "Sentiment"]
                
            st.dataframe(df[display_cols].head(10))
    
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        st.stop()
    
    # Word Clouds - only if we have enough data
    try:
        if len(df) > 5:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("☁️ Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                pos_reviews = df[df["Sentiment"] == "Positive"]["Cleaned_Review"].dropna()
                if len(pos_reviews) > 0:
                    pos_text = " ".join(pos_reviews)
                    if len(pos_text) > 50:  # Ensure we have enough text
                        # Smaller word cloud
                        wc_pos = WordCloud(width=300, height=200, background_color="white").generate(pos_text)
                        # RESIZED word cloud
                        fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Positive Reviews")
                    else:
                        st.info("Not enough positive review text for word cloud")
                else:
                    st.info("No positive reviews found")
            
            with col2:
                neg_reviews = df[df["Sentiment"] == "Negative"]["Cleaned_Review"].dropna()
                if len(neg_reviews) > 0:
                    neg_text = " ".join(neg_reviews)
                    if len(neg_text) > 50:  # Ensure we have enough text
                        # Smaller word cloud
                        wc_neg = WordCloud(width=300, height=200, background_color="black", colormap="Reds").generate(neg_text)
                        # RESIZED word cloud
                        fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Negative Reviews")
                    else:
                        st.info("Not enough negative review text for word cloud")
                else:
                    st.info("No negative reviews found")
    except Exception as e:
        st.warning(f"Error generating word clouds: {str(e)}")
    
    # Top words analysis - simple word frequency analysis
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🔍 Common Words Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Review Keywords")
            pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"])
            
            if pos_words:
                # Create a bar chart with RESIZED dimensions and smaller fonts
                pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                bars = ax.barh(pos_df['Word'][::-1], pos_df['Count'][::-1], color='green')
                
                # Add count values next to each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                            str(int(width)), va='center', fontsize=7)
                            
                ax.set_title("Top Words in Positive Reviews", fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for positive keyword analysis")
        
        with col2:
            st.markdown("#### Negative Review Keywords")
            neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"])
            
            if neg_words:
                # Create a bar chart with RESIZED dimensions and smaller fonts
                neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                bars = ax.barh(neg_df['Word'][::-1], neg_df['Count'][::-1], color='red')
                
                # Add count values next to each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                            str(int(width)), va='center', fontsize=7)
                            
                ax.set_title("Top Words in Negative Reviews", fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for negative keyword analysis")
    except Exception as e:
        st.warning(f"Error in keyword analysis: {str(e)}")
    
    # NEW SECTION: Smart Phrase Analysis
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🧩 Smart Phrase Analysis")
        st.markdown("Common phrases used by customers, ranked by frequency and sentiment")
        
        # Get phrase data
        phrase_data = get_phrase_frequency_data(df)
        
        if phrase_data:
            # Create tabs for positive, negative, and all phrases
            pos_phrases = [p for p in phrase_data if p['sentiment_label'] == 'Positive']
            neg_phrases = [p for p in phrase_data if p['sentiment_label'] == 'Negative']
            
            tabs = st.tabs(["All Phrases", "Positive Phrases", "Negative Phrases"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Visualization of top phrases
                    top_phrases = phrase_data[:10]
                    phrases_df = pd.DataFrame({
                        'Phrase': [p['phrase'] for p in top_phrases],
                        'Count': [p['count'] for p in top_phrases],
                        'Sentiment': [p['sentiment_label'] for p in top_phrases]
                    })
                    
                    # Create horizontal bar chart with colored bars based on sentiment
                    fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"])
                    colors = ['green' if s == 'Positive' else 'red' if s == 'Negative' else 'gray' 
                             for s in phrases_df['Sentiment']]
                    
                    bars = ax.barh(phrases_df['Phrase'][::-1], phrases_df['Count'][::-1], color=colors)
                    
                    # Add count values
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                                str(int(width)), va='center', fontsize=7)
                    
                    ax.set_title("Top Phrases in Reviews", fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Sentiment distribution of top phrases
                    sentiment_distribution = pd.Series([p['sentiment_label'] for p in phrase_data]).value_counts()
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=FIGURE_SIZES["pie"])
                    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                    wedges, texts, autotexts = ax.pie(
                        sentiment_distribution, 
                        labels=sentiment_distribution.index, 
                        autopct='%1.1f%%',
                        colors=[colors.get(x, 'blue') for x in sentiment_distribution.index],
                        textprops={'fontsize': 8}
                    )
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                    
                    ax.set_title("Phrase Sentiment Distribution", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Display phrases with their contexts
                st.markdown("### Top Phrases and Examples")
                for i, phrase_info in enumerate(phrase_data[:10], 1):
                    sentiment_color = "green" if phrase_info['sentiment_label'] == 'Positive' else "red" if phrase_info['sentiment_label'] == 'Negative' else "gray"
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <span class="phrase-header">"{phrase_info['phrase']}"</span> 
                        <span class="phrase-count">({phrase_info['count']} occurrences)</span>
                        <div class="phrase-sentiment" style="color:{sentiment_color}">Sentiment: {phrase_info['sentiment_label']}</div>
                        <div style="margin-top:8px;"><strong>Example contexts:</strong></div>
                        {"<ul>" + "".join([f"<li><em>'{context}'</em></li>" for context in phrase_info['contexts']]) + "</ul>" if phrase_info['contexts'] else "<div>No example contexts found</div>"}
                    </div>
                    """, unsafe_allow_html=True)
            
            with tabs[1]:
                if pos_phrases:
                    st.markdown("### Positive Phrases")
                    for i, phrase_info in enumerate(pos_phrases[:10], 1):
                        st.markdown(f"""
                        <div class="insight-card" style="border-left: 4px solid green;">
                            <span class="phrase-header">"{phrase_info['phrase']}"</span> 
                            <span class="phrase-count">({phrase_info['count']} occurrences)</span>
                            <div style="margin-top:8px;"><strong>Example contexts:</strong></div>
                            {"<ul>" + "".join([f"<li><em>'{context}'</em></li>" for context in phrase_info['contexts']]) + "</ul>" if phrase_info['contexts'] else "<div>No example contexts found</div>"}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No positive phrases identified")
            
            with tabs[2]:
                if neg_phrases:
                    st.markdown("### Negative Phrases")
                    for i, phrase_info in enumerate(neg_phrases[:10], 1):
                        st.markdown(f"""
                        <div class="insight-card" style="border-left: 4px solid red;">
                            <span class="phrase-header">"{phrase_info['phrase']}"</span> 
                            <span class="phrase-count">({phrase_info['count']} occurrences)</span>
                            <div style="margin-top:8px;"><strong>Example contexts:</strong></div>
                            {"<ul>" + "".join([f"<li><em>'{context}'</em></li>" for context in phrase_info['contexts']]) + "</ul>" if phrase_info['contexts'] else "<div>No example contexts found</div>"}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No negative phrases identified")
        else:
            st.info("Not enough review data for phrase analysis")

    except Exception as e:
        st.warning(f"Error in phrase analysis: {str(e)}")
    
    # NEW SECTION: Topic Modeling
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📚 Topic Modeling Analysis")
        st.markdown("Discovering key themes in reviews using LDA (Latent Dirichlet Allocation)")
        
        # Perform topic modeling
        lda_model, dictionary = perform_topic_modeling(df['Cleaned_Review'].dropna().tolist())
        
        if lda_model and dictionary:
            # Number of topics to display
            num_topics = min(5, lda_model.num_topics)
            
            # Visualize topics
            topics_df = pd.DataFrame()
            
            for topic_id in range(num_topics):
                # Get the top words for this topic
                words = [word for word, prob in lda_model.show_topic(topic_id, topn=8)]
                probs = [prob for word, prob in lda_model.show_topic(topic_id, topn=8)]
                
                # Add to the dataframe
                topic_df = pd.DataFrame({
                    'word': words,
                    'probability': probs,
                    'topic': f'Topic {topic_id+1}'
                })
                topics_df = pd.concat([topics_df, topic_df])
            
            # Create horizontal bar chart
            fig = px.bar(
                topics_df, 
                x='probability', 
                y='word', 
                color='topic', 
                facet_col='topic',
                facet_col_wrap=1,  # Stack topics vertically
                height=min(100 * num_topics, 600),
                width=700,
                orientation='h',
                labels={'probability': 'Probability', 'word': 'Word'},
                title='Top Words in Each Topic'
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
            
            # Topic interpretation - manual interpretation of discovered topics
            st.markdown("### Topic Interpretations")
            
            # Initialize topic interpretations (in a real app, these would be dynamically generated)
            topic_interpretations = []
            
            for topic_id in range(num_topics):
                # Get words for this topic
                topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=8)]
                topic_word_str = ", ".join(topic_words)
                
                # Simple heuristic interpretations based on words
                if any(word in topic_words for word in ['service', 'staff', 'friendly', 'helpful']):
                    interpretation = "Customer Service Experience"
                elif any(word in topic_words for word in ['food', 'delicious', 'taste', 'menu', 'dish']):
                    interpretation = "Food Quality and Menu Options"
                elif any(word in topic_words for word in ['clean', 'bathroom', 'cleanliness', 'dirty']):
                    interpretation = "Cleanliness and Hygiene"
                elif any(word in topic_words for word in ['price', 'expensive', 'value', 'worth']):
                    interpretation = "Pricing and Value"
                elif any(word in topic_words for word in ['wait', 'time', 'long', 'slow', 'quick']):
                    interpretation = "Wait Times and Efficiency"
                elif any(word in topic_words for word in ['atmosphere', 'ambiance', 'comfortable', 'environment']):
                    interpretation = "Atmosphere and Ambiance"
                elif any(word in topic_words for word in ['reservation', 'booking', 'table', 'full']):
                    interpretation = "Reservations and Availability"
                elif any(word in topic_words for word in ['delivery', 'order', 'online', 'pickup']):
                    interpretation = "Delivery and Online Ordering"
                else:
                    interpretation = f"Topic {topic_id+1}"
                
                topic_interpretations.append({
                    'topic_id': topic_id,
                    'words': topic_words,
                    'interpretation': interpretation
                })
            
            # Display interpretations
            for topic in topic_interpretations:
                st.markdown(f"""
                <div class="topic-card">
                    <h4>{topic['interpretation']}</h4>
                    <p><strong>Key words:</strong> {', '.join(topic['words'])}</p>
                    <p><em>This topic appears to focus on aspects related to {topic['interpretation'].lower()}</em></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough review data for reliable topic modeling")
    
    except Exception as e:
        st.warning(f"Error in topic modeling: {str(e)}")

    # Truly Smart AI-Powered Recommendations
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🧠 AI-Powered Insights & Recommendations")
        
        # Add new function for AI-powered recommendations
        def generate_ai_recommendations(df, phrase_data=None, topic_data=None):
            """
            Generate AI-powered recommendations based on comprehensive review analysis
            """
            if len(df) < 5:
                return {
                    "strengths": ["Not enough review data to generate AI insights"],
                    "improvements": ["Not enough review data to generate AI insights"],
                    "actions": ["Collect more customer reviews to enable in-depth analysis"]
                }
                
            # 1. Analyze sentiment distribution and trends
            sentiment_counts = df["Sentiment"].value_counts()
            total_reviews = len(df)
            positive_pct = sentiment_counts.get("Positive", 0) / total_reviews * 100
            negative_pct = sentiment_counts.get("Negative", 0) / total_reviews * 100
            neutral_pct = sentiment_counts.get("Neutral", 0) / total_reviews * 100
            
            # Calculate average rating if available
            avg_rating = None
            if "rating_num" in df.columns and df["rating_num"].notna().any():
                avg_rating = df["rating_num"].mean()
            
            # 2. Extract sentiment-specific review content
            positive_reviews = df[df["Sentiment"] == "Positive"]["Cleaned_Review"].dropna().tolist()
            negative_reviews = df[df["Sentiment"] == "Negative"]["Cleaned_Review"].dropna().tolist()
            
            # 3. Find co-occurring topics in negative reviews
            negative_phrases = []
            if phrase_data:
                negative_phrases = [p['phrase'] for p in phrase_data if p['sentiment_label'] == 'Negative']
            
            # 4. Extract themes using NLP
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Prepare for advanced analysis
            all_reviews = df["Cleaned_Review"].dropna().tolist()
            
            # Use TF-IDF to find important terms
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(
                    max_features=100, 
                    stop_words='english', 
                    ngram_range=(1, 3),
                    min_df=2
                )
                tfidf_matrix = tfidf.fit_transform(all_reviews)
                terms = tfidf.get_feature_names_out()
                
                # Get most important terms by TF-IDF score
                from scipy.sparse import csr_matrix
                tfidf_sums = tfidf_matrix.sum(axis=0)
                tfidf_scores = [(term, tfidf_sums[0, idx]) for idx, term in enumerate(terms)]
                top_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:20]
                
                # Split by sentiment
                pos_tfidf = tfidf.transform(positive_reviews) if positive_reviews else None
                neg_tfidf = tfidf.transform(negative_reviews) if negative_reviews else None
                
                # Get sentiment-specific important terms
                pos_terms = []
                neg_terms = []
                
                if pos_tfidf is not None and pos_tfidf.shape[0] > 0:
                    pos_sums = pos_tfidf.sum(axis=0)
                    pos_scores = [(term, pos_sums[0, idx]) for idx, term in enumerate(terms)]
                    pos_terms = sorted(pos_scores, key=lambda x: x[1], reverse=True)[:10]
                
                if neg_tfidf is not None and neg_tfidf.shape[0] > 0:
                    neg_sums = neg_tfidf.sum(axis=0)
                    neg_scores = [(term, neg_sums[0, idx]) for idx, term in enumerate(terms)]
                    neg_terms = sorted(neg_scores, key=lambda x: x[1], reverse=True)[:10]
            except Exception as e:
                # Fallback if advanced NLP fails
                pos_terms = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"], 5)
                neg_terms = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"], 5)
            
            # 5. Correlation analysis
            # Find correlations between ratings and specific terms/phrases
            correlations = []
            if "rating_num" in df.columns and df["rating_num"].notna().any():
                try:
                    # Check if we have phrases from earlier analysis
                    if phrase_data:
                        for phrase_info in phrase_data[:15]:  # Check top phrases
                            phrase = phrase_info['phrase']
                            # Create a temporary column indicating if review contains phrase
                            df['temp_has_phrase'] = df['Cleaned_Review'].str.contains(phrase, case=False, na=False).astype(int)
                            # Calculate correlation with rating
                            corr = df['rating_num'].corr(df['temp_has_phrase'])
                            if not pd.isna(corr) and abs(corr) > 0.15:  # Meaningful correlation threshold
                                correlations.append({
                                    'phrase': phrase,
                                    'correlation': corr,
                                    'sentiment': phrase_info['sentiment_label']
                                })
                            # Clean up temporary column
                            df.drop('temp_has_phrase', axis=1, inplace=True)
                except Exception:
                    pass
            
            # 6. Extract high-impact reviews
            # Find reviews with very low ratings but detailed feedback
            high_impact_reviews = []
            if "rating_num" in df.columns:
                low_rating_detailed = df[(df["rating_num"] <= 2) & (df["Cleaned_Review"].str.len() > 100)]
                if len(low_rating_detailed) > 0:
                    for _, row in low_rating_detailed.head(3).iterrows():
                        high_impact_reviews.append({
                            'rating': row['rating_num'],
                            'snippet': row['snippet'][:150] + "..." if len(row['snippet']) > 150 else row['snippet']
                        })
            
            # 7. Now generate insights based on all this analysis
            insights = {
                "strengths": [],
                "improvements": [],
                "actions": []
            }
            
            # STRENGTH INSIGHTS
            if positive_pct > 70:
                insights["strengths"].append(f"Extremely positive sentiment profile with {positive_pct:.1f}% positive reviews - you're outperforming most businesses in customer satisfaction")
            elif positive_pct > 60:
                insights["strengths"].append(f"Strong positive sentiment with {positive_pct:.1f}% positive reviews - your customers are generally satisfied")
            
            if avg_rating and avg_rating > 4.2:
                insights["strengths"].append(f"Impressive average rating of {avg_rating:.1f}/5.0 - significantly above industry average")
            elif avg_rating and avg_rating > 3.8:
                insights["strengths"].append(f"Solid average rating of {avg_rating:.1f}/5.0 - customers generally rate you favorably")
            
            # Add insights from positive terms/phrases
            if pos_terms:
                # Extract just the terms without scores for readability
                if isinstance(pos_terms[0], tuple):
                    top_pos_terms = [term for term, _ in pos_terms[:5]]
                else:
                    top_pos_terms = [term for term, _ in pos_terms[:5]]
                
                # Analyze the positive terms for themes
                service_terms = ['service', 'staff', 'friendly', 'helpful', 'professional']
                product_terms = ['food', 'quality', 'product', 'delicious', 'taste', 'fresh']
                value_terms = ['value', 'price', 'affordable', 'worth']
                experience_terms = ['atmosphere', 'environment', 'ambiance', 'clean', 'comfortable']
                
                pos_themes = []
                if any(term in top_pos_terms or any(term in pt for pt in top_pos_terms) for term in service_terms):
                    pos_themes.append("customer service")
                if any(term in top_pos_terms or any(term in pt for pt in top_pos_terms) for term in product_terms):
                    pos_themes.append("product quality")
                if any(term in top_pos_terms or any(term in pt for pt in top_pos_terms) for term in value_terms):
                    pos_themes.append("value for money")
                if any(term in top_pos_terms or any(term in pt for pt in top_pos_terms) for term in experience_terms):
                    pos_themes.append("customer experience")
                
                if pos_themes:
                    themes_str = ", ".join(pos_themes[:-1]) + (" and " + pos_themes[-1] if len(pos_themes) > 1 else pos_themes[0])
                    insights["strengths"].append(f"You excel in {themes_str} based on customer feedback analysis")
                
                # Add specific term analysis
                insights["strengths"].append(f"Key positive terms in reviews: {', '.join(top_pos_terms)}")
            
            # Add insights from positive phrases if available
            if phrase_data:
                pos_phrases = [p for p in phrase_data if p['sentiment_label'] == 'Positive'][:3]
                if pos_phrases:
                    phrase_str = ", ".join([f'"{p["phrase"]}"' for p in pos_phrases])
                    insights["strengths"].append(f"Customers frequently praise these aspects: {phrase_str}")
                    
                    # If we have contexts, provide a specific example
                    if any(p['contexts'] for p in pos_phrases):
                        for p in pos_phrases:
                            if p['contexts']:
                                insights["strengths"].append(f'Example positive feedback: "{p["contexts"][0]}"')
                                break
            
            # IMPROVEMENT INSIGHTS
            if negative_pct > 30:
                insights["improvements"].append(f"Concerning level of negative sentiment ({negative_pct:.1f}%) requires immediate attention")
            elif negative_pct > 15:
                insights["improvements"].append(f"Moderate negative sentiment ({negative_pct:.1f}%) suggests room for improvement")
            
            if avg_rating and avg_rating < 3.5:
                insights["improvements"].append(f"Below average rating of {avg_rating:.1f}/5.0 indicates significant customer satisfaction issues")
            elif avg_rating and avg_rating < 4.0:
                insights["improvements"].append(f"Average rating of {avg_rating:.1f}/5.0 has potential for improvement")
            
            # Add insights from negative terms/phrases
            if neg_terms:
                # Extract just the terms without scores for readability
                if isinstance(neg_terms[0], tuple):
                    top_neg_terms = [term for term, _ in neg_terms[:5]]
                else:
                    top_neg_terms = [term for term, _ in neg_terms[:5]]
                
                # Analyze negative terms for themes
                service_issues = ['slow', 'rude', 'wait', 'service', 'staff', 'unprofessional']
                product_issues = ['quality', 'poor', 'bad', 'cold', 'tasteless', 'stale', 'overcooked']
                value_issues = ['expensive', 'overpriced', 'cost', 'price', 'value']
                experience_issues = ['dirty', 'noise', 'crowded', 'uncomfortable', 'ambiance']
                
                neg_themes = []
                if any(term in top_neg_terms or any(term in pt for pt in top_neg_terms) for term in service_issues):
                    neg_themes.append("customer service")
                if any(term in top_neg_terms or any(term in pt for pt in top_neg_terms) for term in product_issues):
                    neg_themes.append("product quality")
                if any(term in top_neg_terms or any(term in pt for pt in top_neg_terms) for term in value_issues):
                    neg_themes.append("value perception")
                if any(term in top_neg_terms or any(term in pt for pt in top_neg_terms) for term in experience_issues):
                    neg_themes.append("facility conditions")
                
                if neg_themes:
                    themes_str = ", ".join(neg_themes[:-1]) + (" and " + neg_themes[-1] if len(neg_themes) > 1 else neg_themes[0])
                    insights["improvements"].append(f"Critical areas needing improvement: {themes_str}")
                
                # Add specific term analysis
                insights["improvements"].append(f"Key negative terms in reviews: {', '.join(top_neg_terms)}")
            
            # Add insights from negative phrases if available
            if phrase_data:
                neg_phrases = [p for p in phrase_data if p['sentiment_label'] == 'Negative'][:3]
                if neg_phrases:
                    phrase_str = ", ".join([f'"{p["phrase"]}"' for p in neg_phrases])
                    insights["improvements"].append(f"Recurring customer complaints focus on: {phrase_str}")
                    
                    # If we have contexts, provide a specific example
                    if any(p['contexts'] for p in neg_phrases):
                        for p in neg_phrases:
                            if p['contexts']:
                                insights["improvements"].append(f'Example negative feedback: "{p["contexts"][0]}"')
                                break
            
            # Add insights from high-impact reviews
            if high_impact_reviews:
                review = high_impact_reviews[0]
                insights["improvements"].append(f'Consider this critical low-rated feedback: "{review["snippet"]}"')
            
            # ACTIONABLE RECOMMENDATIONS
            # Add correlation-based actions
            if correlations:
                for corr in correlations[:2]:
                    if corr['correlation'] > 0:
                        insights["actions"].append(f'Emphasize "{corr["phrase"]}" in your service/product as it strongly correlates with higher ratings')
                    else:
                        insights["actions"].append(f'Address issues related to "{corr["phrase"]}" as it strongly correlates with lower ratings')
            
            # Create industry-specific action items based on themes identified
            if 'customer service' in neg_themes:
                insights["actions"].append("Implement staff training focused on customer interaction and service efficiency")
            if 'product quality' in neg_themes:
                insights["actions"].append("Review product/food quality control processes and implement consistency checks")
            if 'value perception' in neg_themes:
                insights["actions"].append("Evaluate pricing strategy or enhance value communication to address price sensitivity concerns")
            if 'facility conditions' in neg_themes:
                insights["actions"].append("Conduct a thorough facility audit and prioritize cleanliness/comfort improvements")
            
            # Add smart actionable recommendations based on sentiment profile
            if positive_pct < 50:
                insights["actions"].append("Consider conducting focus groups with dissatisfied customers to gain deeper understanding of their concerns")
            
            # Add general strategic recommendations
            insights["actions"].append("Implement a systematic review response strategy, prioritizing detailed responses to negative reviews")
            
            if negative_pct > 20:
                insights["actions"].append("Develop a 90-day improvement plan focusing on the top 3 customer pain points identified in this analysis")
            
            # Return all insights
            return insights
        
        # Generate the AI-powered recommendations
        with st.spinner("Generating AI insights..."):
            recommendations = generate_ai_recommendations(df, phrase_data)
            
            # Display recommendations in an attractive format
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌟 Strengths & Competitive Advantages")
                if recommendations["strengths"]:
                    for i, insight in enumerate(recommendations["strengths"], 1):
                        st.markdown(f"""
                        <div style="background-color:#f0f7ff; border-left:4px solid #2e7d32; padding:15px; margin-bottom:10px; border-radius:5px;">
                            <strong>{i}.</strong> {insight}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data to generate strength insights")
            
            with col2:
                st.markdown("### 🔍 Critical Improvement Areas")
                if recommendations["improvements"]:
                    for i, insight in enumerate(recommendations["improvements"], 1):
                        st.markdown(f"""
                        <div style="background-color:#fff8e1; border-left:4px solid #e65100; padding:15px; margin-bottom:10px; border-radius:5px;">
                            <strong>{i}.</strong> {insight}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data to generate improvement insights")
                
            # Add a section for actionable next steps
            if recommendations["actions"]:
                st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📝 Strategic Action Plan")
                
                for i, action in enumerate(recommendations["actions"], 1):
                    st.markdown(f"""
                    <div style="background-color:#e8f5e9; border-left:4px solid #0277bd; padding:15px; margin-bottom:10px; border-radius:5px;">
                        <strong>Action {i}:</strong> {action}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add implementation timeline suggestion
                st.markdown("### ⏱️ Suggested Implementation Timeline")
                
                timeline_data = {
                    "Immediate (1-2 weeks)": recommendations["actions"][:2] if len(recommendations["actions"]) >= 2 else recommendations["actions"],
                    "Short-term (1 month)": recommendations["actions"][2:4] if len(recommendations["actions"]) >= 4 else recommendations["actions"][2:] if len(recommendations["actions"]) > 2 else [],
                    "Medium-term (2-3 months)": recommendations["actions"][4:] if len(recommendations["actions"]) > 4 else []
                }
                
                for timeframe, actions in timeline_data.items():
                    if actions:
                        st.markdown(f"#### {timeframe}")
                        for action in actions:
                            st.markdown(f"- {action}")
    except Exception as e:
        st.warning(f"Error generating advanced recommendations: {str(e)}")
        # Fallback to simpler analysis if the advanced one fails
        st.markdown("Falling back to basic recommendations due to analysis complexity.")
        
        # Simple recommendations
        try:
            # Basic sentiment analysis
            sentiment_counts = df["Sentiment"].value_counts()
            pos_pct = sentiment_counts.get("Positive", 0) / len(df) * 100 if len(df) > 0 else 0
            neg_pct = sentiment_counts.get("Negative", 0) / len(df) * 100 if len(df) > 0 else 0
            
            # Get top words
            pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"], 5)
            neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"], 5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Strengths")
                st.markdown(f"- {pos_pct:.1f}% of reviews are positive")
                if pos_words:
                    st.markdown(f"- Top positive words: {', '.join([word for word, _ in pos_words])}")
                
            with col2:
                st.markdown("### Areas for Improvement")
                st.markdown(f"- {neg_pct:.1f}% of reviews are negative")
                if neg_words:
                    st.markdown(f"- Top negative words: {', '.join([word for word, _ in neg_words])}")
        except Exception as e:
            st.error(f"Could not generate recommendations: {str(e)}")
    
    # Download Results
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📎 Download Your Results")
        
        if "reviews_df" in st.session_state and st.session_state.reviews_df is not None:
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download Reviews CSV",
                data=csv,
                file_name="review_analysis.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.warning(f"Error with download functionality: {str(e)}")

else:
    # Show a placeholder or example when no data is loaded
    if place_id:
        st.info("Click 'Fetch & Analyze Reviews' to start the analysis.")
    else:
        st.info("Enter a Google Place ID and click 'Fetch & Analyze Reviews' to start.")
        
        # Show an example of what the app does
        st.subheader("📱 App Features")
        st.markdown("""
        - 🤖 **Automated Review Analysis**: Get insights from hundreds of reviews in seconds
        - 📈 **Sentiment Timeline**: Track how sentiment changes over time
        - 🔍 **Common Words Analysis**: Discover what words customers mention most often
        - ☁️ **Word Clouds**: Visualize common words in positive and negative reviews
        - 📊 **Sentiment Analysis**: AI-powered sentiment detection using VADER
        - 🧩 **Smart Phrase Analysis**: Extract meaningful phrases and their contexts
        - 📚 **Topic Modeling**: Discover hidden themes in your customer reviews
        - 🌍 **Multi-language Support**: Analyze reviews in Swahili and other languages
        - 💡 **Smart Recommendations**: Get actionable business advice based on customer feedback
        """)

# Requirements for this application
# pip install streamlit pandas nltk matplotlib wordcloud serpapi gensim spacy textblob scikit-learn googletrans==4.0.0-rc1 plotly seaborn
