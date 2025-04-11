import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Step 1: Load and Sample Data
# -----------------------------
print("📥 Loading cleaned dataset...")
df = pd.read_csv("data/cleaned_zoom_reviews.csv")
df["at"] = pd.to_datetime(df["at"], errors="coerce")
df = df.sort_values("at")

# Sample 5,000 reviews across time (stratified by month)
print("🔄 Sampling 5,000 reviews across time...")
sampled_df = df.groupby(df["at"].dt.to_period("M")).apply(
    lambda x: x.sample(min(150, len(x)), random_state=42)
).reset_index(drop=True)

# Combine text + emojis
sampled_df["combined"] = sampled_df["content"].astype(str) + " " + sampled_df["emojis"].fillna("")

# -----------------------------
# Step 2: Load HuggingFace Model
# -----------------------------
print("🤗 Loading HuggingFace RoBERTa model...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# -----------------------------
# Step 3: Run Sentiment Analysis
# -----------------------------
def classify_sentiment(text):
    try:
        result = sentiment_pipe(text[:512])[0]
        return result["label"].lower()
    except:
        return "neutral"

print("🧠 Running sentiment classification on sampled reviews...")
sampled_df["hf_sentiment"] = sampled_df["combined"].apply(classify_sentiment)

# -----------------------------
# Step 4: Map Model Labels
# -----------------------------
label_map = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "neutral": "neutral"
}
sampled_df["hf_sentiment_label"] = sampled_df["hf_sentiment"].map(label_map)

# -----------------------------
# Step 5: Save Updated Data
# -----------------------------
os.makedirs("data", exist_ok=True)
output_path = "data/zoom_with_hf_sentiment_sampled.csv"
sampled_df.to_csv(output_path, index=False)
print(f"✅ Sentiment data saved to: {output_path}")

# -----------------------------
# Step 6: Visualizations
# -----------------------------
print("📊 Creating sentiment visualizations...")
os.makedirs("output/figures", exist_ok=True)

# Bar Chart
plt.figure(figsize=(6, 4))
sns.countplot(x="hf_sentiment_label", data=sampled_df, palette="Set2", order=["positive", "neutral", "negative"])
plt.title("RoBERTa Sentiment Distribution (Sampled)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/figures/roberta_sentiment_bar.png")
plt.close()

# Pie Chart
plt.figure(figsize=(6, 6))
sampled_df["hf_sentiment_label"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightgreen", "lightgray", "salmon"])
plt.title("RoBERTa Sentiment Proportion")
plt.ylabel("")
plt.tight_layout()
plt.savefig("output/figures/roberta_sentiment_pie.png")
plt.close()

# Sentiment Over Time
monthly_sentiment = sampled_df.groupby(sampled_df["at"].dt.to_period("M"))["hf_sentiment_label"].value_counts().unstack().fillna(0)
monthly_sentiment.index = monthly_sentiment.index.to_timestamp()
monthly_sentiment.plot(kind="line", figsize=(12, 6), marker="o")
plt.title("Monthly RoBERTa Sentiment Trends")
plt.xlabel("Month")
plt.ylabel("Review Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/figures/roberta_sentiment_over_time.png")
plt.close()

print("✅ All plots saved in: output/figures/")
