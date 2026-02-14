# ============================================================================
# Performs the full analysis pipeline:
#   1. Load cleaned tweet data
#   2. Compute VADER sentiment scores for each tweet
#   3. Identify the top 50 most common words in the dataset
#   4. Categorize tweets into themes using keyword dictionaries
#   5. Generate summary statistics and plots by theme
#   6. Conduct one-sided Welch's t-test (safety vs access sentiment)
#   7. Run OLS linear regression with covariates
#   8. Save all outputs (tables, plots, test results) to the OUTPUT folder
#
# Prerequisites:
#   - Run preprocessing.py first to generate the cleaned CSV
#   - Install packages: pip install pandas vaderSentiment scipy statsmodels matplotlib seaborn
# ============================================================================

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so the script runs without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05  # significance level for the hypothesis test

# 1. Load Cleaned Data
print("Step 1 - Loading cleaned data...")
df = pd.read_csv(DATA_DIR / "covid19_vaccine_tweets_cleaned.csv")
print(f"  Loaded {len(df)} tweets with columns: {list(df.columns)}")

# 2. Compute VADER Sentiment Scores
#    VADER produces four scores per text: positive, negative, neutral, and a
#    normalized compound score in [-1, 1]. The compound score is the primary
#    metric used in our analysis.
print("\nStep 2 - Computing VADER sentiment scores...")
analyzer = SentimentIntensityAnalyzer()

# Apply VADER to every tweet
vader_results = df["tweet_text"].astype(str).apply(analyzer.polarity_scores)
df["vader_compound"] = vader_results.apply(lambda d: d["compound"])
df["vader_pos"]      = vader_results.apply(lambda d: d["pos"])
df["vader_neg"]      = vader_results.apply(lambda d: d["neg"])
df["vader_neu"]      = vader_results.apply(lambda d: d["neu"])

print(f"  Compound score summary:")
print(f"    Mean:   {df['vader_compound'].mean():.4f}")
print(f"    Median: {df['vader_compound'].median():.4f}")
print(f"    Std:    {df['vader_compound'].std():.4f}")

# 3. Compute Tweet Length (character count) - used as a covariate later
df["tweet_length"] = df["tweet_text"].astype(str).apply(len)

# 4. Identify Top 50 Most Common Words
#    We tokenise on whitespace, remove common English stopwords, and count.
print("\nStep 3 - Identifying top 50 most common words...")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "am", "s", "t", "d", "ll", "ve", "re", "don", "didn",
    "doesn", "hasn", "haven", "isn", "wasn", "weren", "won", "wouldn",
    "couldn", "shouldn", "also", "get", "got", "said", "says", "say",
    "new", "one", "two", "first", "second", "now", "even", "still",
    "like", "since", "back", "well", "much", "going", "right", "us",
    "make", "made", "know", "take", "want", "see", "come", "go", "way",
    "need", "many", "any", "let", "put", "thing", "things",
    "day", "time", "year", "years", "people", "last", "long", "great",
    "old", "big", "small", "good", "bad", "best", "part", "high", "low",
    "think", "it's", "don't", "i'm", "he's", "she's", "we're",
    "they're", "i've", ""
}

all_words = []
for text in df["tweet_text"].dropna():
    all_words.extend(str(text).split())

word_counts = Counter(w for w in all_words if w not in STOPWORDS and len(w) > 1)
top_50 = word_counts.most_common(50)

# Save to CSV
top_words_df = pd.DataFrame(top_50, columns=["word", "count"])
top_words_df.to_csv(OUTPUT_DIR / "top_50_words.csv", index=False)
print("  Top 50 words saved to output/top_50_words.csv")
for w, c in top_50[:15]:
    print(f"    {w}: {c}")

# 5. Define Theme Keyword Dictionaries
#    Per the MI2 analysis plan, we categorize the top 50 most common words
#    into the four themes below. Only words from the top-50 list are used.
#    Words that are geographic (russia, india, canada, etc.) or non-semantic
#    (&amp;, ??) are excluded. Brand names are tracked separately as a covariate.

# Safety / side-effects: top-50 words relating to risk, harm, or concern
safety_keywords = ["side", "against", "cases", "lockdown"]

# Access / appointments: top-50 words relating to getting vaccinated
access_keywords = ["dose", "doses", "shot", "2nd", "1st", "received", "getting", "jab"]

# Eligibility: no top-50 words are eligibility-specific, so this theme
# will have zero matches; we keep it for completeness per the plan.
eligibility_keywords = []

# General information: top-50 words relating to vaccines, covid, or health news
general_keywords = [
    "vaccine", "vaccines", "vaccination", "vaccinated",
    "covid19", "covid", "coronavirus", "covid-19",
    "covidvaccine", "covid19vaccine",
    "health", "effective", "approved", "million", "use",
]

# Vaccine-brand keywords (used as a covariate, not a theme)
brand_keywords = [
    "moderna", "covaxin", "sputnikv", "pfizerbiontech", "pfizer",
    "astrazeneca", "oxfordastrazeneca", "sinopharm", "sinovac",
    "covishield", "sputnik",
]

# 6. Categorize Each Tweet into Themes
#    A tweet can belong to more than one theme.
print("\nStep 4 - Categorizing tweets by theme...")

def has_theme(text, keywords):
    """Return True if any keyword appears as a whole word in text."""
    words = set(str(text).split())
    return any(kw in words for kw in keywords)

df["theme_safety"]      = df["tweet_text"].apply(lambda x: has_theme(x, safety_keywords))
df["theme_access"]      = df["tweet_text"].apply(lambda x: has_theme(x, access_keywords))
df["theme_eligibility"] = df["tweet_text"].apply(lambda x: has_theme(x, eligibility_keywords))
df["theme_general"]     = df["tweet_text"].apply(lambda x: has_theme(x, general_keywords))
df["brand_mention"]     = df["tweet_text"].apply(lambda x: has_theme(x, brand_keywords))

theme_map = {
    "Safety/Side Effects": "theme_safety",
    "Access/Appointments": "theme_access",
    "Eligibility":         "theme_eligibility",
    "General Information": "theme_general",
}

for name, col in theme_map.items():
    n = df[col].sum()
    print(f"  {name}: {n} tweets ({n/len(df)*100:.1f}%)")
print(f"  Brand Mention: {df['brand_mention'].sum()} tweets "
      f"({df['brand_mention'].mean()*100:.1f}%)")

# 7. Summary Statistics by Theme
print("\nStep 5 - Computing summary statistics by theme...")

summary_rows = []
for theme_name, col in theme_map.items():
    subset = df[df[col]]
    if len(subset) == 0:
        print(f"\n  {theme_name}: 0 tweets (skipped)")
        continue
    row = {
        "Theme": theme_name,
        "N": len(subset),
        "Mean Compound": round(subset["vader_compound"].mean(), 4),
        "Std Compound":  round(subset["vader_compound"].std(), 4),
        "Median Compound": round(subset["vader_compound"].median(), 4),
        "% Negative (< -0.05)": round((subset["vader_compound"] < -0.05).mean() * 100, 1),
        "% Neutral":            round(((subset["vader_compound"] >= -0.05) &
                                       (subset["vader_compound"] <= 0.05)).mean() * 100, 1),
        "% Positive (> 0.05)":  round((subset["vader_compound"] > 0.05).mean() * 100, 1),
        "Mean Tweet Length": round(subset["tweet_length"].mean(), 1),
    }
    summary_rows.append(row)
    print(f"\n  {theme_name} (n={row['N']}):")
    print(f"    Mean compound:  {row['Mean Compound']}")
    print(f"    Std compound:   {row['Std Compound']}")
    print(f"    Median compound:{row['Median Compound']}")
    print(f"    % Negative:     {row['% Negative (< -0.05)']}%")
    print(f"    % Neutral:      {row['% Neutral']}%")
    print(f"    % Positive:     {row['% Positive (> 0.05)']}%")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "summary_statistics_by_theme.csv", index=False)
print("\n  Saved: output/summary_statistics_by_theme.csv")

# 8. Plots
print("\nStep 6 - Generating plots...")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# --- Plot 1: Overlapping histograms (safety vs access) ---
safety_scores = df.loc[df["theme_safety"], "vader_compound"]
access_scores = df.loc[df["theme_access"], "vader_compound"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(safety_scores, bins=40, alpha=0.6, density=True,
        label=f"Safety/Side Effects (n={len(safety_scores)})", color="#FF6B6B")
ax.hist(access_scores, bins=40, alpha=0.6, density=True,
        label=f"Access/Appointments (n={len(access_scores)})", color="#4ECDC4")
ax.axvline(x=safety_scores.mean(), color="#CC0000", linestyle="--", linewidth=1.5,
           label=f"Safety mean = {safety_scores.mean():.3f}")
ax.axvline(x=access_scores.mean(), color="#008888", linestyle="--", linewidth=1.5,
           label=f"Access mean = {access_scores.mean():.3f}")
ax.set_xlabel("VADER Compound Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("VADER Compound Sentiment: Safety vs Access Tweets", fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution_safety_vs_access.png")
plt.close()
print("  Saved: sentiment_distribution_safety_vs_access.png")

# --- Plot 2: Box plot of compound score across themes with data ---
# Skip themes with 0 tweets (eligibility has no top-50 keywords)
active_themes = {k: v for k, v in theme_map.items() if df[v].sum() > 0}
fig, ax = plt.subplots(figsize=(10, 6))
theme_data   = [df.loc[df[col], "vader_compound"] for col in active_themes.values()]
theme_labels = list(active_themes.keys())
colors = ["#FF6B6B", "#4ECDC4", "#96CEB4"]
bp = ax.boxplot(theme_data, labels=theme_labels, patch_artist=True,
                boxprops=dict(alpha=0.7))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_ylabel("VADER Compound Score", fontsize=12)
ax.set_title("VADER Compound Sentiment by Tweet Theme", fontsize=14)
ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot_sentiment_by_theme.png")
plt.close()
print("  Saved: boxplot_sentiment_by_theme.png")

# --- Plot 3: Bar chart of mean compound by theme ---
fig, ax = plt.subplots(figsize=(8, 5))
means = [df.loc[df[col], "vader_compound"].mean() for col in active_themes.values()]
bars = ax.bar(list(active_themes.keys()), means, color=colors, alpha=0.8, edgecolor="black")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Mean VADER Compound Score", fontsize=12)
ax.set_title("Mean VADER Compound Sentiment by Theme", fontsize=14)
ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_sentiment_by_theme.png")
plt.close()
print("  Saved: mean_sentiment_by_theme.png")

# 9. Hypothesis Test - One-Sided Welch's t-test
#    H0: mu_safety >= mu_access   (safety NOT more negative)
#    H1: mu_safety <  mu_access   (safety IS more negative)
print("\nStep 7 - One-Sided Welch's t-test (safety vs access)...")

t_stat, p_two = stats.ttest_ind(safety_scores, access_scores, equal_var=False)

# Convert two-sided p to one-sided (testing safety < access)
if t_stat < 0:
    p_one = p_two / 2
else:
    p_one = 1 - p_two / 2

reject = p_one < ALPHA

print(f"  Safety mean:  {safety_scores.mean():.4f}  (n={len(safety_scores)})")
print(f"  Access mean:  {access_scores.mean():.4f}  (n={len(access_scores)})")
print(f"  Difference:   {safety_scores.mean() - access_scores.mean():.4f}")
print(f"  t-statistic:  {t_stat:.4f}")
print(f"  p-value (one-sided): {p_one:.6f}")
if reject:
    print(f"  --> REJECT H0 at alpha={ALPHA}. Safety tweets are significantly more negative.")
else:
    print(f"  --> FAIL TO REJECT H0 at alpha={ALPHA}.")

# Save results to text file
with open(OUTPUT_DIR / "hypothesis_test_results.txt", "w") as f:
    f.write("One-Sided Welch's Two-Sample t-Test\n")
    f.write("=" * 50 + "\n\n")
    f.write("H0: mu(safety) >= mu(access)\n")
    f.write("    Safety-related tweets are NOT more negative than access-related tweets.\n")
    f.write("H1: mu(safety) <  mu(access)\n")
    f.write("    Safety-related tweets have significantly more negative sentiment.\n\n")
    f.write(f"Significance level: alpha = {ALPHA}\n\n")
    f.write(f"Safety tweets  (n = {len(safety_scores)}):\n")
    f.write(f"  Mean VADER compound: {safety_scores.mean():.4f}\n")
    f.write(f"  Std:                 {safety_scores.std():.4f}\n\n")
    f.write(f"Access tweets  (n = {len(access_scores)}):\n")
    f.write(f"  Mean VADER compound: {access_scores.mean():.4f}\n")
    f.write(f"  Std:                 {access_scores.std():.4f}\n\n")
    f.write(f"t-statistic:         {t_stat:.4f}\n")
    f.write(f"p-value (one-sided): {p_one:.6f}\n\n")
    if reject:
        f.write(f"Result: REJECT H0 (p = {p_one:.6f} < {ALPHA})\n")
        f.write("Safety-themed tweets have significantly more negative sentiment "
                "than access-themed tweets.\n")
    else:
        f.write(f"Result: FAIL TO REJECT H0 (p = {p_one:.6f} >= {ALPHA})\n")
        f.write("Insufficient evidence that safety-themed tweets are more negative "
                "than access-themed tweets.\n")
print("  Saved: output/hypothesis_test_results.txt")

# 10. Linear Regression with Covariates
#     Outcome: VADER compound score
#     Predictors: is_safety (binary), tweet_length, brand_mention (binary)
print("\nStep 8 - OLS Linear Regression...")

reg_df = df[["vader_compound", "theme_safety", "tweet_length", "brand_mention"]].copy()
reg_df["theme_safety"]  = reg_df["theme_safety"].astype(int)
reg_df["brand_mention"] = reg_df["brand_mention"].astype(int)

X = sm.add_constant(reg_df[["theme_safety", "tweet_length", "brand_mention"]])
y = reg_df["vader_compound"]

model = sm.OLS(y, X).fit()
print(model.summary())

# Save regression output
with open(OUTPUT_DIR / "regression_summary.txt", "w") as f:
    f.write(str(model.summary()))
    f.write("\n\n--- Key Numbers ---\n")
    f.write(f"R-squared:          {model.rsquared:.4f}\n")
    f.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}\n\n")
    f.write("Coefficients:\n")
    for name, coef, pval in zip(model.params.index, model.params, model.pvalues):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        f.write(f"  {name:20s}  coef={coef:+.6f}  p={pval:.6f} {sig}\n")
print("  Saved: output/regression_summary.txt")

# 11. Save Final Analyzed Dataset
df.to_csv(DATA_DIR / "covid19_vaccine_tweets_analyzed.csv", index=False)
print(f"\nFinal analyzed dataset saved to data/covid19_vaccine_tweets_analyzed.csv")

print("\n" + "=" * 50)
print("Analysis complete. All outputs are in the output/ folder.")
print("=" * 50)
