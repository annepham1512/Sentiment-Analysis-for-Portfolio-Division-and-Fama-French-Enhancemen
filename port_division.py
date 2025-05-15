import os
import pandas as pd
import numpy as np

# Define paths
base_path = './data/sentiment'
output_folder = './data/portfolios'
os.makedirs(output_folder, exist_ok=True)

# Full list of 88 tickers (already complete)
all_tickers = [
    # Electric Vehicle (EV) Companies - 23
    "TSLA", "BYDDY", "LI", "NIO", "RIVN", "LCID", "XPEV", "NKLA", "PSNY", "GM",
    "F", "VWAGY", "BAMXF", "HYMTF", "KIMTF", "POAHY", "MBGYY", "STLA", "GELYF",
    "GWLLY", "SAIC", "HYLN", "GNZUF",

    # Remaining EV tickers
    "TATAMOTORS.NS", "MAHMF", "RNLSY", "NSANY", "MMTOF",

    # Financial & Banking Companies - 29
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
    "TD", "SCHW", "BK", "AXP", "HSBC", "CFG", "FITB", "MTB", "HBAN",
    "ALLY", "KEY", "RY", "SAN", "NTRS", "RF", "SYF", "NBHC", "ZION", "FHN",

    # Technology Companies - 36
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSM", "ADBE", "INTC",
    "CSCO", "ORCL", "IBM", "CRM", "QCOM", "AVGO", "TXN", "AMD", "AMAT", "MU",
    "NET", "NOW", "SNOW", "DOCU", "SHOP", "UBER", "LYFT", "SNAP", "HRB", "DDOG"
]

# Dictionary mapping ticker -> company name
ticker_to_name = {
    "TSLA": "Tesla", "BYDDY": "BYD Company", "LI": "Li Ideal", "NIO": "NIO",
    "RIVN": "Rivian", "LCID": "Lucid Group", "XPEV": "XPeng", "NKLA": "Nikola",
    "PSNY": "Polestar", "GM": "General Motors", "F": "Ford", "VWAGY": "Volkswagen",
    "BAMXF": "BMW", "HYMTF": "Hyundai", "KIMTF": "Kia", "POAHY": "Porsche",
    "MBGYY": "Mercedes-Benz", "STLA": "Stellantis", "GELYF": "Geely",
    "GWLLY": "Great Wall Motors", "SAIC": "SAIC Motor", "HYLN": "Hyliion",
    "GNZUF": "GAC Group", "TATAMOTORS.NS": "Tata Motors", "MAHMF": "Mahindra",
    "RNLSY": "Renault", "NSANY": "Nissan", "MMTOF": "Mitsubishi Motors",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "C": "Citigroup", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "USB": "U.S. Bancorp",
    "PNC": "PNC Financial", "TFC": "Truist Financial", "COF": "Capital One",
    "TD": "TD Bank", "SCHW": "Charles Schwab", "BK": "Bank of New York Mellon",
    "AXP": "American Express", "HSBC": "HSBC",
    "CFG": "Citizens Financial", "FITB": "Fifth Third Bank", "MTB": "M&T Bank",
    "HBAN": "Huntington Bancshares", "ALLY": "Ally Financial", "KEY": "KeyCorp",
    "RY": "Royal Bank of Canada", "SAN": "Santander", "NTRS": "Northern Trust",
    "RF": "Regions Financial", "SYF": "Synchrony Financial", "NBHC": "National Bank Holdings",
    "ZION": "Zions Bancorporation", "FHN": "First Horizon",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "GOOG": "Google",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA", "TSM": "TSMC",
    "ADBE": "Adobe", "INTC": "Intel", "CSCO": "Cisco", "ORCL": "Oracle",
    "IBM": "IBM", "CRM": "Salesforce", "QCOM": "Qualcomm", "AVGO": "Broadcom",
    "TXN": "Texas Instruments", "AMD": "AMD", "AMAT": "Applied Materials",
    "MU": "Micron", "NET": "Cloudflare", "NOW": "ServiceNow", "SNOW": "Snowflake",
    "DOCU": "DocuSign", "SHOP": "Shopify", "UBER": "Uber", "LYFT": "Lyft",
    "SNAP": "Snap", "HRB": "H&R Block", "DDOG": "Datadog"
}

# Dictionary to store average sentiment per ticker and category
company_sentiments = {}

# Read AI, ESG, and General sentiments
for ticker in all_tickers:
    ai_file = os.path.join(base_path, 'ai', f"{ticker}_ai.csv")
    esg_file = os.path.join(base_path, 'esg', f"{ticker}_esg.csv")
    general_file = os.path.join(base_path, 'general', f"{ticker}_general.csv")

    ai_score = pd.read_csv(ai_file)['sentiment_score'].mean() if os.path.exists(ai_file) else np.nan
    esg_score = pd.read_csv(esg_file)['sentiment_score'].mean() if os.path.exists(esg_file) else np.nan
    general_score = pd.read_csv(general_file)['sentiment_score'].mean() if os.path.exists(general_file) else np.nan

    company_sentiments[ticker] = {
        'ai': ai_score,
        'esg': esg_score,
        'general': general_score
    }

# Convert to DataFrame
sentiment_df = pd.DataFrame.from_dict(company_sentiments, orient='index').reset_index()
sentiment_df.columns = ['ticker', 'ai', 'esg', 'general']

# Fill NaN with 0 where we have no data
sentiment_df[['ai', 'esg', 'general']] = sentiment_df[['ai', 'esg', 'general']].fillna(0)

# Create composite scores
sentiment_df['AI+General'] = (sentiment_df['ai'] + sentiment_df['general']) / 2
sentiment_df['ESG+General'] = (sentiment_df['esg'] + sentiment_df['general']) / 2
sentiment_df['Combined'] = (sentiment_df['ai'] + sentiment_df['esg'] + sentiment_df['general']) / 3

# Function to divide into top/middle/bottom third
def categorize_evenly(df, column):
    df_sorted = df.sort_values(by=column, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    third = n // 3

    df_sorted['category'] = 'Neutral'
    df_sorted.loc[:third - 1, 'category'] = 'Positive'
    df_sorted.loc[third:2 * third - 1, 'category'] = 'Neutral'
    df_sorted.loc[2 * third:, 'category'] = 'Negative'

    return df_sorted[['ticker', 'category']]

# Portfolio types
portfolio_types = ['AI+General', 'ESG+General', 'Combined']
final_output = []

for col in portfolio_types:
    categorized = categorize_evenly(sentiment_df, col)
    categorized['name'] = categorized['ticker'].map(ticker_to_name)
    categorized['display'] = categorized.apply(lambda row: f"{row['ticker']}({row['name']})", axis=1)

    grouped = categorized.groupby('category')['display'].apply(list).reindex(['Positive', 'Neutral', 'Negative'], fill_value=[])

    for category, companies in grouped.items():
        final_output.append({
            'Portfolio': col,
            'Category': category,
            'Companies': ', '.join(companies)
        })

# Convert to DataFrame and save
final_df = pd.DataFrame(final_output, columns=['Portfolio', 'Category', 'Companies'])

# Save to CSV
output_file = os.path.join(output_folder, 'portfolios_summary.csv')
final_df.to_csv(output_file, index=False)
print(f"✅ Portfolios saved to {output_file}")