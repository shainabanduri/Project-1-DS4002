import pandas as pd #need to pip install pandas if not in venv
import re
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Build path to data file
data_path = BASE_DIR / "data" / "archive" / "covid-19_vaccine_tweets_with_sentiment.csv"

df = pd.read_csv(data_path, encoding="latin1")

# drop rows with null tweets in twwet_text section
df = df.dropna(subset=["tweet_text"])

#convert to lowercase
df["tweet_text"] = df["tweet_text"].str.lower()

#remove URLs
df["tweet_text"] = df["tweet_text"].apply(
    lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x)
)

#remove @mentions
df["tweet_text"] = df["tweet_text"].apply(
    lambda x: re.sub(r"@\w+", "", x)
)

#remove hashtag symbols but keep the text of the hashtags
df["tweet_text"] = df["tweet_text"].str.replace("#", "", regex=False)

#remove extra whitespace
df["tweet_text"] = df["tweet_text"].apply(
    lambda x: re.sub(r"\s+", " ", x).strip()
)

#save cleaned data
df.to_csv(BASE_DIR / "data" / "archive" /"covid19_vaccine_tweets_cleaned.csv", index=False)

print("Preprocessing complete. Cleaned file saved.")
