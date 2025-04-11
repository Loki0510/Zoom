import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud

# Load data
st.title("Zoom Emoji Sentiment Explorer")
df = pd.read_csv('data/zoom_with_emoji_sentiment.csv')

# Sidebar filter
st.sidebar.header('Filter')
version_options = df['reviewCreatedVersion'].dropna().unique().tolist()
selected_version = st.sidebar.selectbox('Select App Version:', version_options)

# Filter data by version
filtered_df = df[df['reviewCreatedVersion'] == selected_version].copy()

# Sentiment summary
st.subheader("Sentiment Summary (Text vs Emoji)")
sentiment_labels = ['positive', 'neutral', 'negative']
review_counts = [filtered_df['hf_sentiment_label'].tolist().count(label) for label in sentiment_labels]
emoji_counts = [filtered_df['emoji_sentiment'].tolist().count(label) for label in sentiment_labels]

col1, col2 = st.columns(2)
with col1:
    st.metric("Text - Positive", review_counts[0])
    st.metric("Text - Neutral", review_counts[1])
    st.metric("Text - Negative", review_counts[2])

with col2:
    st.metric("Emoji - Positive", emoji_counts[0])
    st.metric("Emoji - Neutral", emoji_counts[1])
    st.metric("Emoji - Negative", emoji_counts[2])

# Sentiment agreement summary
agreement_df = filtered_df[['hf_sentiment_label', 'emoji_sentiment']].dropna()
agreement_df['Agreement'] = agreement_df['hf_sentiment_label'] == agreement_df['emoji_sentiment']
total_reviews = agreement_df.shape[0]
total_agree = agreement_df['Agreement'].sum()
agree_percent = round((total_agree / total_reviews) * 100, 2) if total_reviews > 0 else 0

st.subheader("Sentiment Agreement Overview")
st.metric(label="Agreement %", value=f"{agree_percent}%", delta=f"{total_agree}/{total_reviews} reviews")

# Sentiment distribution comparison
st.header('Sentiment Distribution Comparison')
sentiment_comparison = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'Review Content': review_counts,
    'Emoji Sentiment': emoji_counts
})

sentiment_comparison_melt = sentiment_comparison.melt(
    id_vars='Sentiment',
    value_vars=['Review Content', 'Emoji Sentiment'],
    var_name='Sentiment Type',
    value_name='Count'
)

fig = px.bar(
    sentiment_comparison_melt,
    x='Sentiment', y='Count', color='Sentiment Type',
    barmode='stack', text='Count',
    title="Review vs Emoji Sentiment Distribution"
)
fig.update_traces(texttemplate='%{text}', textposition='inside')
st.plotly_chart(fig)

# Top emojis distribution
st.header("Top Emojis by Frequency")
emoji_count = filtered_df['emojis'].explode().value_counts().reset_index()
emoji_count.columns = ['Emoji', 'Count']
fig = px.bar(emoji_count.head(20), x='Emoji', y='Count', title="Top 20 Emojis Used")
fig.update_layout(xaxis_tickfont_size=20)
st.plotly_chart(fig)

# Sentiment over time
st.header('Sentiment Over Time')
filtered_df['Month'] = pd.to_datetime(filtered_df['at']).dt.to_period('M').astype(str)
monthly_sentiment_all = filtered_df.groupby(['Month', 'hf_sentiment_label']).size().reset_index(name='count')
fig = px.line(
    monthly_sentiment_all,
    x='Month', y='count', color='hf_sentiment_label',
    markers=True, title="Monthly Text Sentiment Trends"
)
st.plotly_chart(fig)

# Sentiment match/mismatch filter
st.header('Sentiment Match vs Mismatch')
match_filter = st.selectbox('Filter by Sentiment Match:', ['All', 'Match', 'Mismatch'])

if match_filter == 'Match':
    filtered_df = filtered_df[filtered_df['hf_sentiment_label'] == filtered_df['emoji_sentiment']]
elif match_filter == 'Mismatch':
    filtered_df = filtered_df[filtered_df['hf_sentiment_label'] != filtered_df['emoji_sentiment']]

# Sentiment-specific filter
sentiment_filter = st.selectbox("Filter by Sentiment:", ['All', 'positive', 'neutral', 'negative'])
if sentiment_filter != 'All':
    filtered_df = filtered_df[filtered_df['hf_sentiment_label'] == sentiment_filter]

# Display filtered reviews
st.subheader("Filtered Reviews")
st.dataframe(filtered_df[['content', 'hf_sentiment_label', 'emoji_sentiment']])

# Word Cloud for mismatches
if match_filter == 'Mismatch' and not filtered_df.empty:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df['content'].dropna()))
    st.image(wordcloud.to_array(), use_column_width=True, caption="Word Cloud from Mismatched Reviews")
