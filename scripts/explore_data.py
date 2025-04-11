import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import emoji
from matplotlib import font_manager

# Load updated dataset
df = pd.read_csv("data/cleaned_zoom_reviews.csv")
df["at"] = pd.to_datetime(df["at"])
df["appVersion"] = df["appVersion"].fillna("unknown")

# Plot 1: Review count over time (monthly)
df["month"] = df["at"].dt.to_period("M")
review_count = df.groupby("month").size()

plt.figure(figsize=(12, 6))
review_count.plot(kind="bar", color="skyblue")
plt.title("Review Count Over Time (Monthly)")
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

# Plot 2: Rating distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="score", data=df, palette="viridis")
plt.title("Star Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Load font that supports emojis
emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"  # Segoe UI Emoji
emoji_font = font_manager.FontProperties(fname=emoji_font_path)

# Count top emojis
emoji_series = df["emojis"].dropna()
emoji_counter = Counter("".join(emoji_series))
top_emojis = emoji_counter.most_common(20)

# Prepare data
emojis, counts = zip(*top_emojis)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(emojis), y=list(counts), color="skyblue")
plt.title("Top 20 Most Frequent Emojis", fontproperties=emoji_font, fontsize=16)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontproperties=emoji_font, fontsize=16)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save output

plt.show()

# Plot 4: Word Cloud of reviews
text_blob = " ".join(df["content"].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_blob)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Reviews")
plt.show()
