import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import emoji
import os

# Initialize HuggingFace RoBERTa model for sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Emoji Sentiment Mapping (expand as necessary)
emoji_sentiment_map = {
    "😊": "positive", "😍": "positive", "😢": "negative", "😡": "negative", "😐": "neutral",
    "😂": "neutral", "😭": "negative", "❤️": "positive", "👍": "positive", "👎": "negative",
    "😜": "positive", "😎": "positive", "😱": "negative", "😃": "positive", "😕": "neutral",
    "😩": "negative"
}

# Function to extract emojis from a text string
def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

# Function to classify emoji sentiment
def classify_emoji_sentiment(emojis):
    sentiment = "neutral"  # Default sentiment if no match
    for emoji_char in emojis:
        if emoji_char in emoji_sentiment_map:
            sentiment = emoji_sentiment_map[emoji_char]
            break
    return sentiment

# Function to get sentiment from the text using RoBERTa
def get_sentiment(texts, batch_size=8):
    try:
        # Ensure all inputs are strings and handle missing values
        texts = [str(text) if text is not None else "" for text in texts]

        # Initialize progress bar using tqdm
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Process text in batches
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
            
            # Get sentiment results from the model
            batch_results = sentiment_pipe(batch)

            # Extract sentiment labels
            labels = [result['label'].lower() for result in batch_results]  # Extract the sentiment label
            results.extend(labels)
        
        return results
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return ["neutral"] * len(texts)

# Main function to perform sentiment analysis
def analyze_sentiment(df):
    # Ensure all text inputs are strings and handle NaN
    df["content"] = df["content"].fillna("").astype(str)
    
    # Step 1: Apply RoBERTa sentiment analysis on review text
    df["text_sentiment"] = get_sentiment(df["content"])
    
    # Step 2: Extract emojis and classify emoji sentiment
    df["emojis"] = df["content"].apply(extract_emojis)
    df["emoji_sentiment"] = df["emojis"].apply(lambda x: classify_emoji_sentiment(x) if isinstance(x, str) else "neutral")
    
    # Combine results: Optionally, you can also analyze whether the sentiments match or not
    df["sentiment_match"] = df["text_sentiment"] == df["emoji_sentiment"]
    
    return df

# Load and analyze Zoom data
df = pd.read_csv("data/cleaned_zoom_reviews.csv")
df = analyze_sentiment(df)

# Save the result
os.makedirs("data", exist_ok=True)
df.to_csv("data/zoom_with_hf_and_emoji_sentiment.csv", index=False)

print(f"Sentiment analysis complete and saved to: data/zoom_with_hf_and_emoji_sentiment.csv")
