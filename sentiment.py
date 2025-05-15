import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from gnews import GNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define company tickers by sector
ev_companies = {
    "TSLA": "Tesla", "BYDDY": "BYD Company", "LI": "Li Ideal", "NIO": "NIO",
    "RIVN": "Rivian", "LCID": "Lucid Group", "XPEV": "XPeng", "NKLA": "Nikola",
    "PSNY": "Polestar", "GM": "General Motors", "F": "Ford", "VWAGY": "Volkswagen",
    "BAMXF": "BMW", "HYMTF": "Hyundai", "KIMTF": "Kia", "POAHY": "Porsche",
    "MBGYY": "Mercedes-Benz", "STLA": "Stellantis", "GELYF": "Geely",
    "GWLLY": "Great Wall Motors", "SAIC": "SAIC Motor", "HYLN": "Hyliion",
    "GNZUF": "GAC Group", "TATAMOTORS.NS": "Tata Motors", "MAHMF": "Mahindra",
    "RNLSY": "Renault", "NSANY": "Nissan", "MMTOF": "Mitsubishi Motors"
}

finbank_companies = {
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "C": "Citigroup", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "USB": "U.S. Bancorp",
    "PNC": "PNC Financial", "TFC": "Truist Financial", "COF": "Capital One",
    "TD": "TD Bank", "SCHW": "Charles Schwab", "BK": "Bank of New York Mellon",
    "AXP": "American Express", "HSBC": "HSBC",
    "CFG": "Citizens Financial", "FITB": "Fifth Third Bank", "MTB": "M&T Bank",
    "HBAN": "Huntington Bancshares", "ALLY": "Ally Financial", "KEY": "KeyCorp",
    "RY": "Royal Bank of Canada", "SAN": "Santander", "NTRS": "Northern Trust",
    "RF": "Regions Financial", "SYF": "Synchrony Financial", "NBHC": "National Bank Holdings",
    "ZION": "Zions Bancorporation", "FHN": "First Horizon"
}

tech_companies = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "GOOG": "Google",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA", "TSM": "TSMC",
    "ADBE": "Adobe", "INTC": "Intel", "CSCO": "Cisco", "ORCL": "Oracle",
    "IBM": "IBM", "CRM": "Salesforce", "QCOM": "Qualcomm", "AVGO": "Broadcom",
    "TXN": "Texas Instruments", "AMD": "AMD", "AMAT": "Applied Materials",
    "MU": "Micron", "NET": "Cloudflare", "NOW": "ServiceNow", "SNOW": "Snowflake",
    "DOCU": "DocuSign", "SHOP": "Shopify", "UBER": "Uber", "LYFT": "Lyft",
    "SNAP": "Snap", "HRB": "H&R Block", "DDOG": "Datadog"
}

# Merge all companies
all_companies = {**ev_companies, **finbank_companies, **tech_companies}

# Sector mapping
sector_mapping = {}
for ticker in ev_companies:
    sector_mapping[ticker] = "EV"
for ticker in finbank_companies:
    sector_mapping[ticker] = "Financial"
for ticker in tech_companies:
    sector_mapping[ticker] = "Technology"

# Keywords to filter articles
ai_keywords = [
    'AI', 'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'robotic', 'autonomous', 'NLP'
]

esg_keywords = [
    'ESG', 'sustainability', 'carbon footprint', 'climate change',
    'renewable energy', 'social responsibility', 'governance', 'diversity',
    'inclusion', 'CSR', 'corporate social responsibility', 'emissions', 'green',
    'environment', 'net zero', 'clean energy', 'ethics'
]

# Load FinBERT for sentiment analysis
try:
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
except Exception as e:
    logger.warning(f"Error loading FinBERT: {e}")
    tokenizer, model = None, None


def scrape_company_news(ticker, company_name, search_term=""):
    """Scrape news articles related to a company using its ticker or name."""
    logger.info(f"\n🔄 Processing {ticker} - {company_name}")
    results = []

    google_news = GNews(language='en', country='US', max_results=100)

    # Build query
    if search_term:
        query = f"(\"{ticker}\" OR \"{company_name}\") stock {search_term}"
    else:
        query = f"(\"{ticker}\" OR \"{company_name}\") stock"

    try:
        logger.info(f"🔍 Fetching news for: '{query}'")
        articles = google_news.get_news(query)
        logger.info(f"📄 Retrieved {len(articles)} articles")

        for article in articles:
            title = article['title']
            summary = article['description']
            pub_date = article['published date']
            full_text = f"{title}. {summary}"
            sentiment = get_finbert_sentiment(full_text)

            results.append({
                'ticker': ticker,
                'company_name': company_name,
                'title': title,
                'summary': summary,
                'text': full_text,
                'search_term': search_term,
                'sentiment_score': sentiment['score'],
                'sentiment_label': sentiment['label'],
                'published': pub_date
            })

    except Exception as e:
        logger.warning(f"⚠️ Error fetching news for {query}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        df = df[df['published'].notna()]
        df.sort_values('published', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"📅 Found {len(df)} valid articles between {df['published'].min()} and {df['published'].max()}")
    else:
        logger.warning(f"🚫 No articles found for {ticker}")

    return df


def contains_keywords(text, keywords):
    """Check if text contains any of the specified keywords."""
    if not isinstance(text, str):
        return False
    return any(kw.lower() in text.lower() for kw in keywords)


def save_by_type(df, ticker, folder_type):
    """Save news DataFrame to CSV under appropriate folder."""
    base_path = "./data/sentiment"
    os.makedirs(os.path.join(base_path, folder_type), exist_ok=True)
    output_path = os.path.join(base_path, folder_type, f"{ticker}_{folder_type}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Saved {output_path} | Articles: {len(df)}")


def process_and_save(ticker, company_name):
    """Process general, AI, and ESG-related news for a company and save them."""
    logger.info(f"\n🔄 Processing {ticker} - {company_name}")

    # General news
    general_df = scrape_company_news(ticker, company_name)
    if not general_df.empty:
        save_by_type(general_df, ticker, 'general')

    # AI news
    ai_df = scrape_company_news(ticker, company_name, "AI")
    if not ai_df.empty:
        ai_df = ai_df[ai_df.apply(lambda row: contains_keywords(row['text'], ai_keywords), axis=1)]
        save_by_type(ai_df, ticker, 'ai')

    # ESG news
    esg_df = scrape_company_news(ticker, company_name, "ESG")
    if not esg_df.empty:
        esg_df = esg_df[esg_df.apply(lambda row: contains_keywords(row['text'], esg_keywords), axis=1)]
        save_by_type(esg_df, ticker, 'esg')

    return pd.concat([general_df, ai_df, esg_df], ignore_index=True)


def get_finbert_sentiment(text):
    """Analyze sentiment using FinBERT model."""
    if not text or not isinstance(text, str) or not tokenizer or not model:
        return {'score': 0.0, 'label': 'neutral'}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
    labels = ['negative', 'neutral', 'positive']
    label_id = scores.argmax()
    return {
        'score': scores[label_id] if labels[label_id] != 'negative' else -scores[label_id],
        'label': labels[label_id]
    }


def scrape_all_news(companies):
    """Scrape news for all companies and combine into one DataFrame."""
    all_news = []
    for ticker, company_name in companies.items():
        df = process_and_save(ticker, company_name)
        if not df.empty:
            all_news.append(df)
    return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()


def get_rolling_sentiment(news_df, current_date, keywords=None, window_days=60):
    """
    Calculate average sentiment score over a rolling window of `window_days`.
    Returns 0.0 if no matching news articles are found.
    """
    current_date = pd.to_datetime(current_date)
    start_window = current_date - timedelta(days=window_days)

    filtered = news_df[
        (news_df['published'] <= current_date) &
        (news_df['published'] >= start_window)
    ]

    # Apply keyword filtering if specified
    if keywords:
        filtered = filtered[filtered.apply(lambda row: contains_keywords(row['text'], keywords), axis=1)]

    # Return average sentiment score or neutral 0.0 if no matching articles
    if not filtered.empty:
        return filtered['sentiment_score'].mean()
    return 0.0


def generate_daily_sentiment(news_df, start_date="2020-07-01", end_date=None):
    """
    Generate daily sentiment scores using a rolling 60-day window.
    """
    end_date = end_date or datetime.today().strftime('%Y-%m-%d')
    dates = pd.date_range(start=start_date, end=end_date)
    daily_data = []

    for date in dates:
        general_sentiment = get_rolling_sentiment(news_df, date, window_days=60)
        ai_sentiment = get_rolling_sentiment(news_df, date, keywords=ai_keywords, window_days=60)
        esg_sentiment = get_rolling_sentiment(news_df, date, keywords=esg_keywords, window_days=60)

        daily_data.append({
            'Date': date,
            'general_sentiment': general_sentiment,
            'ai_sentiment': ai_sentiment,
            'esg_sentiment': esg_sentiment
        })

    return pd.DataFrame(daily_data).set_index('Date')


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    logger.info("Step 1: Scraping News Articles...")
    news_df = scrape_all_news(all_companies)

    # Save combined news data
    news_df['published'] = pd.to_datetime(news_df['published'], utc=True).dt.date
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    news_df.to_csv(f"./data/sentiment/company_news_{timestamp}.csv", index=False)
    logger.info("✅ News scraping completed.")

    logger.info("Step 2: Generating Daily Sentiment Scores...")
    daily_sentiment_df = generate_daily_sentiment(news_df)
    daily_sentiment_df.to_csv(f"./data/sentiment/daily_sentiment_{timestamp}.csv")
    logger.info("✅ Daily sentiment saved.")