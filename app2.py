import pandas as pd
import streamlit as st
import plotly.express as px

# Load the data (update with the correct path to the CSV file)
df = pd.read_csv('data/zoom_with_emoji_sentiment.csv')

# Sidebar for version selection
st.sidebar.header('Filter by Version')
version_options = df['reviewCreatedVersion'].unique().tolist()
selected_version = st.sidebar.selectbox('Select App Version:', version_options)

# Filter dataset based on selected version
filtered_df = df[df['reviewCreatedVersion'] == selected_version]

# --- Sentiment Distribution Comparison (Review vs Emoji) ---
st.header('Sentiment Distribution Comparison (Review vs Emoji)')
sentiment_comparison = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'Review Content': [
        filtered_df[filtered_df['hf_sentiment_label'] == 'positive'].shape[0],
        filtered_df[filtered_df['hf_sentiment_label'] == 'neutral'].shape[0],
        filtered_df[filtered_df['hf_sentiment_label'] == 'negative'].shape[0],
    ],
    'Emoji Sentiment': [
        filtered_df[filtered_df['emoji_sentiment'] == 'positive'].shape[0],
        filtered_df[filtered_df['emoji_sentiment'] == 'neutral'].shape[0],
        filtered_df[filtered_df['emoji_sentiment'] == 'negative'].shape[0],
    ]
})

sentiment_comparison_melt = sentiment_comparison.melt(
    id_vars='Sentiment',
    value_vars=['Review Content', 'Emoji Sentiment'],
    var_name='Sentiment Type',
    value_name='Count'
)

# Stacked bar chart for sentiment comparison
fig = px.bar(
    sentiment_comparison_melt,
    x='Sentiment',
    y='Count',
    color='Sentiment Type',
    title="Comparison of Review Content vs Emoji Sentiment",
    labels={'Sentiment': 'Sentiment Type', 'Count': 'Number of Reviews'},
    barmode='stack',  # Stack bars for better comparison
    text='Count'  # Add count to the bars for clarity
)

fig.update_traces(texttemplate='%{text}', textposition='inside', insidetextanchor='middle')
st.plotly_chart(fig, key="sentiment_distribution_comparison")

# --- Sentiment Agreement (Emoji vs Review) ---
st.header('Sentiment Agreement (Emoji vs Review)')
agreement_df = filtered_df[['hf_sentiment_label', 'emoji_sentiment']].dropna()
agreement_df['Agreement'] = agreement_df['hf_sentiment_label'] == agreement_df['emoji_sentiment']

# Sankey diagram for sentiment agreement
fig = px.sunburst(
    agreement_df,
    path=['hf_sentiment_label', 'emoji_sentiment'],
    title="Sentiment Agreement: Text vs. Emoji"
)
st.plotly_chart(fig, key="sentiment_agreement")

# --- Emoji Distribution Plot ---
st.header('Top Emojis and Their Sentiment')
emoji_count = filtered_df['emojis'].explode().value_counts().reset_index()
emoji_count.columns = ['Emoji', 'Count']

# Map sentiment to emojis
emoji_sentiment_map = {
    'positive': 'positive',
    'neutral': 'neutral',
    'negative': 'negative'
}
emoji_count['Sentiment'] = emoji_count['Emoji'].map(lambda emoji: emoji_sentiment_map.get(emoji, 'neutral'))

# Bar chart for emoji distribution
fig = px.bar(
    emoji_count,
    x='Emoji',
    y='Count',
    color='Sentiment',
    title="Top Emojis and Their Sentiment",
    labels={'Emoji': 'Emoji', 'Count': 'Frequency'}
)

st.plotly_chart(fig, key="emoji_distribution")

# --- Sentiment Over Time (Frustration Indicator) ---
st.header('Sentiment Over Time')

# Convert 'Month' to string format to handle Plotly serialization
filtered_df['Month'] = pd.to_datetime(filtered_df['at']).dt.to_period('M').astype(str)

# Debug: Check how many records are available for the selected version
st.write(f"Number of records for the selected version: {filtered_df.shape[0]}")

# Group by month and calculate sentiment counts
monthly_sentiment = filtered_df.groupby('Month').agg(
    text_sentiment_count=('hf_sentiment_label', lambda x: x.value_counts().get('positive', 0)),
    emoji_sentiment_count=('emoji_sentiment', lambda x: x.value_counts().get('positive', 0))
).reset_index()

# Debug: Show the aggregated data to verify it looks correct
st.write("Aggregated sentiment data over time:")
st.write(monthly_sentiment)

# Check if there's any data to plot
if monthly_sentiment.empty:
    st.warning("No data available for the selected version or sentiment data.")
else:
    # Plot the line chart for sentiment over time
    fig = px.line(
        monthly_sentiment,
        x='Month',  # Use the string-formatted Month column
        y=['text_sentiment_count', 'emoji_sentiment_count'],
        title="Sentiment Over Time (Text vs Emoji)",
        labels={'text_sentiment_count': 'Text Sentiment Count', 'emoji_sentiment_count': 'Emoji Sentiment Count'}
    )

    st.plotly_chart(fig, key="sentiment_over_time")

# --- Additional Features: Sentiment Match vs Mismatch ---
st.header('Sentiment Match vs Mismatch')
match_filter = st.selectbox('Filter by Sentiment Match:', ['All', 'Match', 'Mismatch'])

if match_filter == 'Match':
    filtered_df = filtered_df[filtered_df['hf_sentiment_label'] == filtered_df['emoji_sentiment']]
elif match_filter == 'Mismatch':
    filtered_df = filtered_df[filtered_df['hf_sentiment_label'] != filtered_df['emoji_sentiment']]

# Display filtered reviews
st.dataframe(filtered_df[['content', 'hf_sentiment_label', 'emoji_sentiment']])

# --- Word Cloud for Mismatched Reviews ---
if match_filter == 'Mismatch':
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df['content']))
    st.image(wordcloud.to_array(), use_column_width=True)
