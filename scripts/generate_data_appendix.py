# ============================================================================
# Generates a Data Appendix PDF following the TIER Protocol 4.0 format.
# Includes: variable definitions, missing data counts, full summary stats
# (mean, std, min, 25th, median, 75th, max) with histograms for quantitative
# variables, and frequency tables with bar charts for categorical variables.
#
# Prerequisites:
#   - Run preprocessing.py and analysis.py first
#   - Install packages: pip install matplotlib pandas seaborn
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "data_appendix.pdf"

# Load all three datasets
df_raw = pd.read_csv(DATA_DIR / "covid-19_vaccine_tweets_with_sentiment.csv", encoding="latin1")
df_clean = pd.read_csv(DATA_DIR / "covid19_vaccine_tweets_cleaned.csv")
df = pd.read_csv(DATA_DIR / "covid19_vaccine_tweets_analyzed.csv")

# ---- Helpers ----

def text_page(pdf, lines, fontsize=9.5):
    """Render lines of text as a PDF page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=fontsize, verticalalignment="top", fontfamily="monospace")
    pdf.savefig(fig)
    plt.close(fig)

def quant_stats(series):
    """Return formatted summary-stat lines for a quantitative variable."""
    n = len(series)
    m = int(series.isna().sum())
    desc = series.describe()
    return [
        f"    Observations: {n}   Missing: {m}",
        f"    Mean:   {desc['mean']:.4f}",
        f"    Std:    {desc['std']:.4f}",
        f"    Min:    {desc['min']:.4f}",
        f"    25th:   {desc['25%']:.4f}",
        f"    Median: {desc['50%']:.4f}",
        f"    75th:   {desc['75%']:.4f}",
        f"    Max:    {desc['max']:.4f}",
    ]

def hist_page(pdf, series, title, xlabel, color="#4ECDC4"):
    """Render a histogram on its own PDF page."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=40, color=color, edgecolor="black", alpha=0.75)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def bar_page(pdf, labels, counts, title, xlabel, color="#4ECDC4"):
    """Render a bar chart on its own PDF page."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=color, edgecolor="black", alpha=0.75)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(c), ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(OUTPUT_PATH) as pdf:

    # ==== TITLE PAGE ====
    text_page(pdf, [
        "DATA APPENDIX",
        "=" * 60,
        "",
        "Project: Sentiment Shifts in Vaccine-Related Tweets by Theme",
        "Group:   Model Citizens",
        "Members: Shaina Banduri, Neil Parikh, Nishana Dahal",
        "Course:  DS 4002",
        "Date:    Feb. 2026",
        "",
        "This appendix documents every variable in the datasets",
        "used in this project, following the TIER Protocol 4.0.",
        "",
        "Datasets documented:",
        "  1. covid-19_vaccine_tweets_with_sentiment.csv (raw)",
        "  2. covid19_vaccine_tweets_cleaned.csv         (preprocessed)",
        "  3. covid19_vaccine_tweets_analyzed.csv         (final analysis)",
    ])

    # ==== DATASET 1: RAW ====
    text_page(pdf, [
        "=" * 60,
        "DATASET 1: Raw Data",
        "File: covid-19_vaccine_tweets_with_sentiment.csv",
        f"Rows: {len(df_raw):,}   Columns: {len(df_raw.columns)}",
        "",
        "Unit of observation: One tweet from Twitter related to",
        "COVID-19 vaccines, with a human-annotated sentiment label.",
        "",
        "--- Variable: tweet_id (float64) ---",
        "  Unique numerical ID for each tweet.",
        "  Source: Original Kaggle dataset.",
        f"  Observations: {len(df_raw)}   Missing: {int(df_raw['tweet_id'].isna().sum())}",
        "",
        "--- Variable: label (int64) ---",
        "  Human-annotated sentiment label.",
        "  Values: 1 = Negative, 2 = Neutral, 3 = Positive.",
        "  Source: Original Kaggle dataset (human annotators).",
        f"  Observations: {len(df_raw)}   Missing: {int(df_raw['label'].isna().sum())}",
        f"  Distribution:",
        f"    1 (Negative): {(df_raw['label']==1).sum()}",
        f"    2 (Neutral):  {(df_raw['label']==2).sum()}",
        f"    3 (Positive): {(df_raw['label']==3).sum()}",
        "",
        "--- Variable: tweet_text (string) ---",
        "  Full text content of the tweet including hashtags,",
        "  URLs, and @mentions.",
        "  Source: Original Kaggle dataset.",
        f"  Observations: {len(df_raw)}   Missing: {int(df_raw['tweet_text'].isna().sum())}",
    ])

    # Bar chart for raw label distribution
    lbl_counts_raw = df_raw["label"].value_counts().sort_index()
    bar_page(pdf, ["Negative (1)", "Neutral (2)", "Positive (3)"],
             lbl_counts_raw.values.tolist(),
             "Raw Data: Distribution of Sentiment Labels", "Label",
             color="#FF6B6B")

    # ==== DATASET 2: CLEANED ====
    text_page(pdf, [
        "=" * 60,
        "DATASET 2: Cleaned Data",
        "File: covid19_vaccine_tweets_cleaned.csv",
        f"Rows: {len(df_clean):,}   Columns: {len(df_clean.columns)}",
        "",
        "Unit of observation: One cleaned tweet. Rows with missing",
        "tweet_text were removed during preprocessing. Text was",
        "lowercased, URLs removed, @mentions removed, # symbols",
        "stripped (hashtag text preserved), whitespace normalized.",
        "",
        "--- Variable: tweet_id (float64) ---",
        "  Same as raw data. Unique tweet identifier.",
        f"  Observations: {len(df_clean)}   Missing: {int(df_clean['tweet_id'].isna().sum())}",
        "",
        "--- Variable: label (int64) ---",
        "  Same as raw data.",
        f"  Observations: {len(df_clean)}   Missing: {int(df_clean['label'].isna().sum())}",
        f"  Distribution:",
        f"    1 (Negative): {(df_clean['label']==1).sum()} ({(df_clean['label']==1).mean()*100:.1f}%)",
        f"    2 (Neutral):  {(df_clean['label']==2).sum()} ({(df_clean['label']==2).mean()*100:.1f}%)",
        f"    3 (Positive): {(df_clean['label']==3).sum()} ({(df_clean['label']==3).mean()*100:.1f}%)",
        "",
        "--- Variable: tweet_text (string) ---",
        "  Cleaned tweet text. All lowercase, no URLs, no @mentions,",
        "  no # symbols (hashtag text preserved), single-spaced.",
        f"  Observations: {len(df_clean)}   Missing: {int(df_clean['tweet_text'].isna().sum())}",
    ])

    # Bar chart for cleaned label distribution
    lbl_counts = df_clean["label"].value_counts().sort_index()
    bar_page(pdf, ["Negative (1)", "Neutral (2)", "Positive (3)"],
             lbl_counts.values.tolist(),
             "Cleaned Data: Distribution of Sentiment Labels", "Label",
             color="#4ECDC4")

    # ==== DATASET 3: ANALYZED (text descriptions) ====
    text_page(pdf, [
        "=" * 60,
        "DATASET 3: Analyzed Data",
        "File: covid19_vaccine_tweets_analyzed.csv",
        f"Rows: {len(df):,}   Columns: {len(df.columns)}",
        "",
        "Unit of observation: One cleaned tweet enriched with VADER",
        "sentiment scores and theme/brand indicator flags computed",
        "by analysis.py.",
        "",
        "Original variables (same as cleaned data):",
        "  tweet_id, label, tweet_text -- see Dataset 2 above.",
        "",
        "The following pages document each added variable with",
        "summary statistics and visualizations.",
    ])

    # --- vader_compound ---
    text_page(pdf, [
        "--- Variable: vader_compound (float64) ---",
        "  VADER normalised compound sentiment score.",
        "  Range: [-1, 1]. -1 = most negative, +1 = most positive.",
        "  Source: Computed from tweet_text using vaderSentiment.",
        "",
    ] + quant_stats(df["vader_compound"]))

    hist_page(pdf, df["vader_compound"],
              "Histogram: vader_compound", "VADER Compound Score", "#FF6B6B")

    # --- vader_pos ---
    text_page(pdf, [
        "--- Variable: vader_pos (float64) ---",
        "  Proportion of text tokens with positive sentiment [0, 1].",
        "  Source: Computed from tweet_text using vaderSentiment.",
        "",
    ] + quant_stats(df["vader_pos"]))

    hist_page(pdf, df["vader_pos"],
              "Histogram: vader_pos", "Positive Proportion", "#96CEB4")

    # --- vader_neg ---
    text_page(pdf, [
        "--- Variable: vader_neg (float64) ---",
        "  Proportion of text tokens with negative sentiment [0, 1].",
        "  Source: Computed from tweet_text using vaderSentiment.",
        "",
    ] + quant_stats(df["vader_neg"]))

    hist_page(pdf, df["vader_neg"],
              "Histogram: vader_neg", "Negative Proportion", "#FF6B6B")

    # --- vader_neu ---
    text_page(pdf, [
        "--- Variable: vader_neu (float64) ---",
        "  Proportion of text tokens with neutral sentiment [0, 1].",
        "  Source: Computed from tweet_text using vaderSentiment.",
        "",
    ] + quant_stats(df["vader_neu"]))

    hist_page(pdf, df["vader_neu"],
              "Histogram: vader_neu", "Neutral Proportion", "#45B7D1")

    # --- tweet_length ---
    text_page(pdf, [
        "--- Variable: tweet_length (int64) ---",
        "  Character count of the cleaned tweet text.",
        "  Source: Computed as len(tweet_text) in analysis.py.",
        "",
    ] + quant_stats(df["tweet_length"]))

    hist_page(pdf, df["tweet_length"],
              "Histogram: tweet_length", "Number of Characters", "#4ECDC4")

    # --- Categorical / boolean variables ---
    bool_vars = {
        "theme_safety":      "True if tweet contains a safety/side-effects keyword.\n"
                             "    Source: Keyword matching in analysis.py.",
        "theme_access":      "True if tweet contains an access/appointments keyword.\n"
                             "    Source: Keyword matching in analysis.py.",
        "theme_eligibility": "True if tweet contains an eligibility keyword.\n"
                             "    Source: Keyword matching in analysis.py.",
        "theme_general":     "True if tweet contains a general-information keyword.\n"
                             "    Source: Keyword matching in analysis.py.",
        "brand_mention":     "True if tweet mentions a vaccine brand name.\n"
                             "    Source: Keyword matching in analysis.py.",
    }

    for var, desc in bool_vars.items():
        t_count = int(df[var].sum())
        f_count = int((~df[var]).sum())
        m_count = int(df[var].isna().sum())
        text_page(pdf, [
            f"--- Variable: {var} (bool) ---",
            f"  {desc}",
            f"  Observations: {len(df)}   Missing: {m_count}",
            f"  Frequency table:",
            f"    True:  {t_count} ({t_count/len(df)*100:.1f}%)",
            f"    False: {f_count} ({f_count/len(df)*100:.1f}%)",
        ])

        bar_page(pdf, ["True", "False"], [t_count, f_count],
                 f"Bar Chart: {var}", var,
                 color="#FF6B6B" if "safety" in var else "#4ECDC4")

    # ==== END ====
    text_page(pdf, [
        "=" * 60,
        "END OF DATA APPENDIX",
        "=" * 60,
    ])

print(f"Data Appendix saved to {OUTPUT_PATH}")
