# ============================================================================
# generate_data_appendix.py
# DS 4002 - Project 1: Sentiment Shifts in Vaccine-Related Tweets by Theme
# Group: Model Citizens (Shaina Banduri, Neil Parikh, Nishana Dahal)
#
# This script generates a Data Appendix PDF for the DATA folder.
# The appendix documents every variable in each dataset used in the analysis.
#
# Prerequisites:
#   - Run preprocessing.py and analysis.py first
#   - Install packages: pip install matplotlib pandas
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "data_appendix.pdf"

# Load the analyzed dataset (has all variables)
df = pd.read_csv(DATA_DIR / "covid19_vaccine_tweets_analyzed.csv")

# Helper to add a page of text
def add_text_page(pdf, lines, fontsize=10):
    """Render a list of text lines as a single PDF page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top", fontfamily="monospace", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

with PdfPages(OUTPUT_PATH) as pdf:

    # ---- Page 1: Title & Overview ----
    lines = [
        "DATA APPENDIX",
        "=" * 60,
        "",
        "Project: Sentiment Shifts in Vaccine-Related Tweets by Theme",
        "Group:   Model Citizens",
        "Members: Shaina Banduri, Neil Parikh, Nishana Dahal",
        "Course:  DS 4002",
        "Date:    Feb. 2026",
        "",
        "This appendix documents every variable in the datasets used",
        "in this project. Three CSV files are described below:",
        "",
        "  1. covid-19_vaccine_tweets_with_sentiment.csv  (raw data)",
        "  2. covid19_vaccine_tweets_cleaned.csv           (after preprocessing)",
        "  3. covid19_vaccine_tweets_analyzed.csv           (after analysis)",
        "",
        "=" * 60,
        "",
        "DATASET 1: Raw Data",
        "File: covid-19_vaccine_tweets_with_sentiment.csv",
        f"Rows: 14,151   Columns: 3",
        "",
        "Unit of observation: One tweet from Twitter related to",
        "COVID-19 vaccines, with a human-annotated sentiment label.",
        "",
        "Variables:",
        "",
        "  tweet_id    (float64)",
        "    Unique numerical ID for each tweet.",
        "    Example: 1.360342e+18",
        "",
        "  label       (int64)",
        "    Human-annotated sentiment label.",
        "    Values: 1 = Negative, 2 = Neutral, 3 = Positive",
        "",
        "  tweet_text  (string)",
        "    Full text content of the tweet, including hashtags,",
        "    URLs, and @mentions.",
        "",
    ]
    add_text_page(pdf, lines)

    # ---- Page 2: Cleaned Dataset ----
    lines = [
        "DATASET 2: Cleaned Data",
        "File: covid19_vaccine_tweets_cleaned.csv",
        f"Rows: 6,000   Columns: 3",
        "",
        "Unit of observation: One cleaned tweet. Rows with missing",
        "tweet text were removed. Text was lowercased, URLs removed,",
        "@mentions removed, # symbols stripped, whitespace normalized.",
        "",
        "Variables:",
        "",
        "  tweet_id    (float64)",
        "    Same as raw data. Unique tweet identifier.",
        "",
        "  label       (int64)",
        "    Same as raw data.",
        "    Distribution in cleaned set:",
        f"      1 (Negative): {(df['label']==1).sum()}  "
        f"({(df['label']==1).mean()*100:.1f}%)",
        f"      2 (Neutral):  {(df['label']==2).sum()}  "
        f"({(df['label']==2).mean()*100:.1f}%)",
        f"      3 (Positive): {(df['label']==3).sum()}  "
        f"({(df['label']==3).mean()*100:.1f}%)",
        "",
        "  tweet_text  (string)",
        "    Cleaned tweet text. All lowercase, no URLs, no @mentions,",
        "    no # symbols (hashtag text preserved), single-spaced.",
        "",
    ]
    add_text_page(pdf, lines)

    # ---- Page 3: Analyzed Dataset - Part 1 ----
    lines = [
        "DATASET 3: Analyzed Data",
        "File: covid19_vaccine_tweets_analyzed.csv",
        f"Rows: {len(df)}   Columns: {len(df.columns)}",
        "",
        "Unit of observation: One cleaned tweet enriched with VADER",
        "sentiment scores and theme/brand indicator flags.",
        "",
        "Original variables (same as cleaned data):",
        "  tweet_id, label, tweet_text",
        "",
        "--- Added Variables ---",
        "",
        "  vader_compound  (float64)",
        "    VADER normalised compound sentiment score [-1, 1].",
        "    -1 = most negative, +1 = most positive.",
        f"    Mean:   {df['vader_compound'].mean():.4f}",
        f"    Median: {df['vader_compound'].median():.4f}",
        f"    Std:    {df['vader_compound'].std():.4f}",
        f"    Min:    {df['vader_compound'].min():.4f}",
        f"    Max:    {df['vader_compound'].max():.4f}",
        "",
        "  vader_pos  (float64)",
        "    Proportion of text with positive sentiment [0, 1].",
        f"    Mean: {df['vader_pos'].mean():.4f}",
        "",
        "  vader_neg  (float64)",
        "    Proportion of text with negative sentiment [0, 1].",
        f"    Mean: {df['vader_neg'].mean():.4f}",
        "",
        "  vader_neu  (float64)",
        "    Proportion of text with neutral sentiment [0, 1].",
        f"    Mean: {df['vader_neu'].mean():.4f}",
        "",
        "  tweet_length  (int64)",
        "    Character count of the cleaned tweet text.",
        f"    Mean:   {df['tweet_length'].mean():.1f}",
        f"    Median: {df['tweet_length'].median():.1f}",
        f"    Min:    {df['tweet_length'].min()}",
        f"    Max:    {df['tweet_length'].max()}",
    ]
    add_text_page(pdf, lines)

    # ---- Page 4: Analyzed Dataset - Part 2 (theme flags) ----
    lines = [
        "DATASET 3 (continued): Theme & Brand Indicator Flags",
        "",
        "  theme_safety  (bool)",
        "    True if the tweet contains at least one keyword from",
        "    the safety/side-effects dictionary.",
        f"    True:  {df['theme_safety'].sum()} ({df['theme_safety'].mean()*100:.1f}%)",
        f"    False: {(~df['theme_safety']).sum()} ({(~df['theme_safety']).mean()*100:.1f}%)",
        "",
        "  theme_access  (bool)",
        "    True if the tweet contains at least one keyword from",
        "    the access/appointments dictionary.",
        f"    True:  {df['theme_access'].sum()} ({df['theme_access'].mean()*100:.1f}%)",
        f"    False: {(~df['theme_access']).sum()} ({(~df['theme_access']).mean()*100:.1f}%)",
        "",
        "  theme_eligibility  (bool)",
        "    True if the tweet contains at least one keyword from",
        "    the eligibility dictionary.",
        f"    True:  {df['theme_eligibility'].sum()} ({df['theme_eligibility'].mean()*100:.1f}%)",
        f"    False: {(~df['theme_eligibility']).sum()} ({(~df['theme_eligibility']).mean()*100:.1f}%)",
        "",
        "  theme_general  (bool)",
        "    True if the tweet contains at least one keyword from",
        "    the general-information dictionary.",
        f"    True:  {df['theme_general'].sum()} ({df['theme_general'].mean()*100:.1f}%)",
        f"    False: {(~df['theme_general']).sum()} ({(~df['theme_general']).mean()*100:.1f}%)",
        "",
        "  brand_mention  (bool)",
        "    True if the tweet mentions at least one vaccine brand",
        "    (Pfizer, Moderna, AstraZeneca, Covaxin, etc.).",
        f"    True:  {df['brand_mention'].sum()} ({df['brand_mention'].mean()*100:.1f}%)",
        f"    False: {(~df['brand_mention']).sum()} ({(~df['brand_mention']).mean()*100:.1f}%)",
        "",
        "=" * 60,
        "END OF DATA APPENDIX",
    ]
    add_text_page(pdf, lines)

print(f"Data Appendix saved to {OUTPUT_PATH}")
