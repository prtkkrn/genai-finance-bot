# 💹 GenAI Finance Bot

A **Streamlit-powered financial research assistant** that combines **GenAI + Market Data**:

- 📄 **PDF Q&A** — Upload Annual Reports / Investor Decks, ask natural questions, and get AI-generated bullet answers (RAG with FAISS + OpenRouter LLMs).
- 📊 **Daily Screener** — Enhanced factor model (Value, Quality, Growth, Risk, Momentum) to rank stocks.
- 📰 **Market Wrap** — Daily NIFTY 50 & SENSEX moves with a brief AI-generated summary.

---

## 🚀 Features
- Upload **company PDFs** → AI extracts and answers queries.
- **Semantic search** over reports with FAISS & Sentence Transformers.
- Daily **Top-5 stock picks** from NSE tickers with logos.
- Automated **Market Wrap** refreshed every 24h.
- Polished **Streamlit UI** with tabs, cards, KPIs, and logos.

---

## ⚙️ Setup

### 1️⃣ Clone repo
```bash
git clone https://github.com/<your-username>/genai-finance-bot.git
cd genai-finance-bot
2️⃣ Create virtual environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Add secrets
Create a .env file:

ini
Copy code
OPENAI_API_KEY=your_openrouter_key_here
5️⃣ Run
bash
Copy code
streamlit run app.py