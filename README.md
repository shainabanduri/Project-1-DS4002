# Model Citizens

## Contents of the Repository
At the top level of the repository, the following files and folders are included:

- **README.md** -- This file. Orientation guide covering software requirements, folder structure, and reproduction instructions.
- **LICENSE.md** -- MIT License specifying terms of use and citation.
- **requirements.txt** -- Python package dependencies for reproducing the analysis.
- **scripts/** -- All source code (preprocessing and analysis), executed in numbered order.
- **data/** -- Raw and processed datasets, plus a Data Appendix PDF.
- **output/** -- All generated figures, tables, and result files.

## 1. Software and Platform

| Item | Details |
|------|---------|
| Language | Python 3.9+ |
| IDE | VSCode |
| Platform | Windows |

### Required Packages

Install all dependencies at once with:

```
pip install -r requirements.txt
```

Individual packages used:

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `vaderSentiment` | VADER compound sentiment scoring |
| `scipy` | Welch's t-test |
| `statsmodels` | OLS linear regression |
| `matplotlib` | Plot generation |
| `seaborn` | Plot styling |

## 2. Map of the Documentation

```
Project-1-DS4002/
├── README.md
├── LICENSE.md
├── requirements.txt
├── scripts/
│   ├── preprocessing.py          # Step 1: Cleans raw tweet data
│   ├── analysis.py               # Step 2: VADER scoring, theme tagging,
│   │                              #         t-test, regression, plots
│   └── generate_data_appendix.py # Step 3: Generates Data Appendix PDF
├── data/
│   ├── covid-19_vaccine_tweets_with_sentiment.csv   # Raw data (14,151 tweets)
│   ├── covid19_vaccine_tweets_cleaned.csv            # After preprocessing (6,000 tweets)
│   ├── covid19_vaccine_tweets_analyzed.csv           # After analysis (with VADER scores & themes)
│   └── data_appendix.pdf                             # Variable-level documentation
└── output/
    ├── summary_statistics_by_theme.csv                   # Theme-level summary stats
    ├── top_50_words.csv                                  # 50 most frequent words
    ├── hypothesis_test_results.txt                       # Welch's t-test output
    ├── regression_summary.txt                            # OLS regression output
    ├── sentiment_distribution_safety_vs_access.png       # Histogram: safety vs access
    ├── boxplot_sentiment_by_theme.png                    # Box plot across all themes
    └── mean_sentiment_by_theme.png                       # Bar chart of mean sentiment
```

## 3. Instructions for Reproducing the Results

### Prerequisites

1. Install Python 3.9 or later.
2. Clone this repository:
   ```
   git clone https://github.com/shainabanduri/Project-1-DS4002.git
   cd Project-1-DS4002
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Step 1: Preprocessing

Run the preprocessing script to clean the raw tweet data. This reads `data/covid-19_vaccine_tweets_with_sentiment.csv`, removes null rows, lowercases text, strips URLs, @mentions, and `#` symbols, and writes the cleaned output.

```
python scripts/preprocessing.py
```

**Input:** `data/covid-19_vaccine_tweets_with_sentiment.csv` (14,151 rows)
**Output:** `data/covid19_vaccine_tweets_cleaned.csv` (6,000 rows)

### Step 2: Analysis

Run the analysis script to compute VADER sentiment scores, categorize tweets into themes, generate plots, run the hypothesis test, and fit the regression model.

```
python scripts/analysis.py
```

**Input:** `data/covid19_vaccine_tweets_cleaned.csv`
**Outputs (all saved to `output/`):**
- `summary_statistics_by_theme.csv` -- mean, median, std of VADER compound by theme
- `top_50_words.csv` -- the 50 most frequent words after stopword removal
- `sentiment_distribution_safety_vs_access.png` -- overlapping histogram
- `boxplot_sentiment_by_theme.png` -- box plot of compound scores by theme
- `mean_sentiment_by_theme.png` -- bar chart of mean compound by theme
- `hypothesis_test_results.txt` -- one-sided Welch's t-test (safety vs access)
- `regression_summary.txt` -- OLS regression with covariates

**Additional output:** `data/covid19_vaccine_tweets_analyzed.csv` (final dataset with all computed columns)

### Step 3: Generate Data Appendix

Run the data appendix script to produce the required Data Appendix PDF in the data folder. This must be run after `analysis.py` since it reads the analyzed dataset.

```
python scripts/generate_data_appendix.py
```

**Input:** `data/covid19_vaccine_tweets_analyzed.csv`
**Output:** `data/data_appendix.pdf`

### Verifying Results

After running all three scripts, the `output/` folder should contain 7 files. Key results to check:
- The hypothesis test should show p < 0.05 (safety tweets are significantly more negative than access tweets).
- The regression coefficient for `theme_safety` should be negative and significant.

## References

[1] "Covid-19 Vaccine Tweets with Sentiment Annotation," Kaggle dataset. Available: https://www.kaggle.com/datasets/datasciencetool/covid19-vaccine-tweets-with-sentiment-annotation

[2] World Health Organization Regional Office for Europe, "Infodemics and misinformation negatively affect people's health behaviours -- new WHO review finds," Sep. 1, 2022. Available: https://www.who.int/europe/news/item/01-09-2022-infodemics-and-misinformation-negatively-affect-people-s-health-behaviours--new-who-review-finds

[3] Centers for Disease Control and Prevention, "Strengthen Vaccination Communications (IQIP)," Jun. 18, 2024. Available: https://www.cdc.gov/iqip/hcp/strategies/strengthen-vaccination-communications.html

[4] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 8, no. 1, 2014. Available: https://doi.org/10.1609/icwsm.v8i1.14550
