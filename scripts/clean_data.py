import pandas as pd
import re
import emoji
import os

# Create paths
# Absolute path
input_path = r"S:\Private\lvemula\ASN3\data\Zoom.xlsx"
output_path = r"S:\Private\lvemula\ASN3\data\cleaned_zoom_reviews.csv"


# Load data
df = pd.read_excel(input_path)

# Drop irrelevant columns
df = df.drop(columns=["userImage", "replyContent", "repliedAt"])

# Drop rows with missing content or score
df = df.dropna(subset=["content", "score"])

# Normalize text: lowercase and strip whitespace
df["content"] = df["content"].astype(str).str.lower().str.strip()

# Extract emojis
def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

df["emojis"] = df["content"].apply(extract_emojis)

# Drop duplicates
df = df.drop_duplicates(subset=["reviewId", "content"])

# Step: Fill or Flag Missing Versions
df["appVersion"] = df["appVersion"].fillna("unknown")
df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("unknown")

# Step: Convert 'at' (timestamp) to datetime
df["at"] = pd.to_datetime(df["at"], errors='coerce')


# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("✅ Cleaned dataset saved to:", output_path)
print("Final Shape:", df.shape)
