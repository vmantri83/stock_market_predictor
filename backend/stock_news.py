import requests
from datetime import date, timedelta
import yfinance as yf
from transformers import pipeline

# Force use of PyTorch
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"  # <-- This forces PyTorch
)


# Summarizer from Hugging Face (you don't need an API key for this!)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# News API config
NEWS_API_KEY = "44efc0f086074757b14e33106b31f05d"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# Financial keywords
FINANCE_KEYWORDS = [
    "stock", "stocks", "market", "markets", "finance", "financial", "invest", "investment",
    "investor", "investing", "money", "shares", "trading", "ipo", "merger", "acquisition",
    "revenue", "loss", "gain", "profit", "growth", "decline", "capital", "dividend", "earnings",
    "valuation", "forecast", "guidance", "quarter", "results", "bullish", "bearish", "downgrade",
    "upgrade", "buy", "sell", "price", "target", "volatility", "equity", "liquidity", "margin",
    "nasdaq", "nyse", "s&p", "index", "bond", "interest rate", "inflation", "deflation",
    "federal reserve", "fed", "economic", "layoffs", "cash flow", "balance sheet"
]

# Fetch news
def fetch_financial_news(company_name, days_back=7, api_key=NEWS_API_KEY):
    today = date.today()
    from_date = today - timedelta(days=days_back)

    params = {
        'q': company_name,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': today.strftime('%Y-%m-%d'),
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 20
    }

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])

        filtered_articles = [
            article['description']
            for article in articles
            if article.get('description') and
               any(kw in article['description'].lower() for kw in FINANCE_KEYWORDS) and
               company_name.lower() in article['description'].lower()  # New filter
        ]
        return filtered_articles

    except Exception as e:
        print("Error fetching news:", e)
        return []

# Fetch stock summary
def fetch_stock_summary(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="7d")
        closing = hist['Close'].tolist()
        change = closing[-1] - closing[0] if len(closing) >= 2 else 0
        return f"The stock closed at Rs. {closing[-1]:.2f}. Weekly change: {'up' if change > 0 else 'down'} by Rs. {abs(change):.2f}."
    except Exception as e:
        return "Could not fetch stock data."

# Main function
def stock_health_report(company_name, ticker_symbol):
    print(f"\n📊 Stock Health Report for: {company_name} ({ticker_symbol})\n")

    stock_summary = fetch_stock_summary(ticker_symbol)
    print("💹 Stock Summary:", stock_summary)

    news_descriptions = fetch_financial_news(company_name)
    if news_descriptions:
        combined_news = " ".join(news_descriptions)[:3000]
        summary = summarizer(combined_news, max_length=150, min_length=5, do_sample=False)[0]['summary_text']
        print("\n📰 Financial News Insight:")
        print(summary)
    else:
        print("\nNo finance-related news found.")

def stock_health_report(company_name, ticker_symbol):
    report = {
        "company": company_name,
        "ticker": ticker_symbol,
        "stock_summary": fetch_stock_summary(ticker_symbol),
        "news_summary": ""
    }

    news_descriptions = fetch_financial_news(company_name)
    if news_descriptions:
        combined_news = " ".join(news_descriptions)[:3000]
        summary = summarizer(combined_news, max_length=150, min_length=5, do_sample=False)[0]['summary_text']
        report["news_summary"] = summary
    else:
        report["news_summary"] = "No finance-related news found."

    return report
