from datetime import datetime, timedelta
import urllib.parse
import re

import feedparser
from openai import OpenAI
from textblob import TextBlob

from agents.base_agent import BaseAgent, UserProfile

class NewsSentimentAgent(BaseAgent):
    def __init__(self, user_profile: UserProfile, openai_api_key=None, use_llm=False):
        super().__init__()
        self.user_profile = user_profile
        self.use_llm = use_llm
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    def llm_sentiment_polarity(self, text, model="gpt-4o"):
        if not self.openai_client:
            return 0
        prompt = (
            "Please analyze the following financial news headline and reply with a single number between -1 (very negative), "
            "0 (neutral), and 1 (very positive) indicating sentiment polarity. Reply with the number only. Headline: "
            f"{text}"
        )
        try:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = completion.choices[0].message.content.strip()
            return float(content)
        except Exception as e:
            print(f"OpenAI API error: {e}\nText: {text}")
            return 0

    def evaluate(self, symbol, dt, max_articles=15):
        # Query Google News with the symbol
        query = symbol
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        sentiments = []
        headlines = []
        articles = 0
        start_date = dt - timedelta(days=7)   # inclusive start
        end_date = dt + timedelta(days=1)
        for entry in feed.entries:
            if articles >= max_articles:
                break
            if 'published_parsed' not in entry:
                continue

            pub_date = datetime(*entry.published_parsed[:6])  # Convert to datetime
            if start_date <= pub_date < end_date:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = f"{title} | {summary}".strip()
                if not text:
                    continue
                cleaned = re.sub(r'<[^>]+>', '', text)
                headlines.append(title)
                if self.use_llm:
                    polarity = self.llm_sentiment_polarity(cleaned)
                else:
                    polarity = TextBlob(cleaned).sentiment.polarity
                sentiments.append(polarity)
                articles += 1
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        reasoning = f"Avg news sentiment for {symbol}: {avg_sentiment:.2f} ({len(sentiments)} news items)"
        return {
            "score": avg_sentiment,
            "reasoning": reasoning,
            "headlines": headlines
        }
