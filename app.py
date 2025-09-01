# app.py

# --- Streamlit MUST be configured first ---
import streamlit as st
st.set_page_config(page_title="GenAI Finance Bot", page_icon="💹", layout="wide")

# Standard imports
import os, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from openai import OpenAI
from pypdf import PdfReader
import yfinance as yf

# Embeddings (semantic RAG)
try:
    import faiss  # faiss-cpu (pinned in requirements)
    FAISS_OK = True
except Exception:
    FAISS_OK = False

from sentence_transformers import SentenceTransformer

# =========================
# Secrets & API Key handling
# =========================
def get_api_key() -> str | None:
    # 1) Streamlit Cloud secrets
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    # 2) Local .env (dev)
    load_dotenv(find_dotenv(), override=False)
    return os.getenv("OPENAI_API_KEY")

API_KEY = get_api_key()

# =========================
# Global Styles
# =========================
st.markdown("""
<style>
:root {
  --card-bg: rgba(255,255,255,0.65);
  --card-border: rgba(0,0,0,0.06);
}
.block-container { padding-top: 1.2rem; }
.section-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.08);
  backdrop-filter: blur(4px);
}
.stTextInput > div > div > input,
.stTextArea textarea { border-radius: 12px !important; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: #eef6ff; color:#0958d9; font-weight:600; font-size:12px;
  border: 1px solid #d0e6ff;
}
.kpi {
  display:flex; flex-direction:column; gap:2px;
  padding:10px 12px; border-radius:12px; border:1px solid var(--card-border);
  background: rgba(255,255,255,0.7);
}
.kpi .label { font-size:12px; opacity:0.7; }
.kpi .value { font-size:18px; font-weight:700; }
.hdr { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
.hdr .title { font-size:22px; font-weight:800; line-height:1.1; }
.hdr .sub { opacity:0.7; font-size:13px; }
hr { border: none; border-top: 1px dashed #e9e9e9; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hdr">
  <div style="font-size:40px; line-height:1;">💹</div>
  <div>
    <div class="title">GenAI Finance Bot</div>
    <div class="sub">PDF Q&A • Daily Screener • Market Wrap</div>
  </div>
  <div style="flex:1;"></div>
  <span class="badge">India 🇮🇳</span>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in **Streamlit Secrets** (Cloud) or a local **.env** (dev).")

# =========================
# Utilities
# =========================
def clean_text(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)                        # collapse spaces
    t = re.sub(r"\n{2,}", "\n", t)                       # collapse blank lines
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)               # fix hyphen line-breaks
    t = re.sub(r"\n(?=[a-z])", " ", t)                   # join accidental line breaks mid-sentence
    t = re.sub(r"\s+\|\s+\d+\s+\|\s*", " ", t)           # remove table-like pipes
    t = re.sub(r"\b(Page|PAGE)\s*\d+\b", " ", t)         # remove 'Page 12'
    t = re.sub(r"\s{2,}", " ", t)                        # collapse leftover double spaces
    return t.strip()

def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i:i + size])
        i += step
    return chunks

def build_faiss(chunks, model):
    vecs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index, vecs

def top_k_chunks(query: str, chunks, index, model, k: int = 6):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, k)
    return [chunks[i] for i in I[0]]

def _safe(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return default
        return float(v)
    except Exception:
        return default

def zscore(s: pd.Series):
    s = s.astype(float)
    denom = s.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return (s - s.mean())
    return (s - s.mean()) / denom

def winsorize(s: pd.Series, lower=0.02, upper=0.98):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def factor_score(df: pd.DataFrame) -> pd.Series:
    # Value (cheaper better): PE, PB, EV/EBITDA
    val = -zscore(winsorize(df["PE"])) + -zscore(winsorize(df["PB"])) + -zscore(winsorize(df["EVEBITDA"]))
    # Quality: ROE, ProfitMargin, OperatingMargin
    quality = zscore(winsorize(df["ROE"])) + zscore(winsorize(df["ProfitMargin"])) + zscore(winsorize(df["OperatingMargin"]))
    # Growth: revenueGrowth, earningsGrowth
    growth = zscore(winsorize(df["RevenueGrowth"])) + zscore(winsorize(df["EarningsGrowth"]))
    # Risk: lower D/E preferred
    risk = -zscore(winsorize(df["DebtToEquity"]))
    # Momentum: 30d price change
    momentum = zscore(winsorize(df["Momentum30d"]))
    return 0.30*val + 0.25*quality + 0.20*growth + 0.10*risk + 0.15*momentum

@st.cache_data(ttl=24*60*60)
def get_stock_snapshot(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            t = yf.Ticker(s + ".NS")
            info = t.info
            hist = t.history(period="1mo")
            if len(hist) >= 2:
                momentum = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100
            else:
                momentum = 0.0

            ev_to_ebitda = info.get("enterpriseToEbitda")
            if ev_to_ebitda is None:
                ebitda = info.get("ebitda")
                enterprise_value = info.get("enterpriseValue")
                ev_to_ebitda = (enterprise_value / ebitda) if (enterprise_value and ebitda and ebitda != 0) else None

            rows.append({
                "Symbol": s,
                "Logo": info.get("logo_url", ""),   # logo url from Yahoo
                "PE": _safe(info.get("trailingPE")),
                "PB": _safe(info.get("priceToBook")),
                "EVEBITDA": _safe(ev_to_ebitda),
                "ROE": _safe(info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None),
                "ProfitMargin": _safe(info.get("profitMargins") * 100 if info.get("profitMargins") else None),
                "OperatingMargin": _safe(info.get("operatingMargins") * 100 if info.get("operatingMargins") else None),
                "RevenueGrowth": _safe(info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None),
                "EarningsGrowth": _safe(info.get("earningsGrowth") * 100 if info.get("earningsGrowth") else None),
                "DebtToEquity": _safe(info.get("debtToEquity")),
                "Momentum30d": _safe(momentum),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["PE","PB","EVEBITDA","ROE","ProfitMargin","OperatingMargin","RevenueGrowth","EarningsGrowth","DebtToEquity","Momentum30d"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)

    df["Score"] = factor_score(df)
    return df

@st.cache_data(ttl=24*60*60)
def get_market_wrap():
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN"}
    out = []
    for name, symbol in indices.items():
        try:
            t = yf.Ticker(symbol)
            hist_1d = t.history(period="5d")
            hist_1m = t.history(period="1mo")
            if len(hist_1d) >= 2:
                d_change = (hist_1d["Close"].iloc[-1] - hist_1d["Close"].iloc[-2]) / hist_1d["Close"].iloc[-2] * 100
            else:
                d_change = 0.0
            if len(hist_1m) >= 2:
                m_change = (hist_1m["Close"].iloc[-1] - hist_1m["Close"].iloc[0]) / hist_1m["Close"].iloc[0] * 100
            else:
                m_change = 0.0
            last = float(hist_1d["Close"].iloc[-1]) if len(hist_1d) else None
            out.append({"Index": name, "Last": last, "DayChangePct": float(d_change), "MonthChangePct": float(m_change)})
        except Exception:
            continue
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    return pd.DataFrame(out), ts

def style_screener_table(df: pd.DataFrame):
    # Gradient requires matplotlib; if unavailable, just format + bar
    try:
        import matplotlib  # noqa: F401
        use_gradient = True
    except Exception:
        use_gradient = False

    df2 = df.copy()
    sty = (df2.style
        .format({
            "PE": "{:.2f}", "PB": "{:.2f}", "EVEBITDA": "{:.2f}",
            "ROE": "{:.1f}%", "ProfitMargin": "{:.1f}%", "OperatingMargin": "{:.1f}%",
            "RevenueGrowth": "{:.1f}%", "EarningsGrowth": "{:.1f}%",
            "DebtToEquity": "{:.2f}", "Momentum30d": "{:.1f}%", "Score": "{:.2f}"
        })
        .bar(subset=["Score"], color="#e8f3ff")
    )
    if use_gradient:
        sty = sty.background_gradient(
            subset=["ROE","ProfitMargin","OperatingMargin","RevenueGrowth","EarningsGrowth","Momentum30d","Score"],
            cmap="Greens"
        )
    return sty

def kpi(label: str, value: str):
    st.markdown(f"""
    <div class="kpi">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# External Services
# =========================
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🧠 PDF Q&A", "📊 Screener", "📰 Market Wrap"])

# =========================
# Tab 1: PDF Q&A
# =========================
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 📄 Upload a company PDF")
        pdf = st.file_uploader("Integrated/Annual Report, Investor Deck, etc. (PDF)", type=["pdf"])
        if pdf is not None:
            try:
                reader = PdfReader(pdf)
                pages = [page.extract_text() or "" for page in reader.pages]
                text = clean_text("\n".join(pages))
                st.success(f"Text extracted • {len(reader.pages)} pages • {len(text):,} chars")

                chunks = chunk_text(text)
                if not FAISS_OK:
                    st.warning("FAISS not available; semantic search is disabled on this platform.")
                else:
                    index, _ = build_faiss(chunks, embed_model)
                    st.session_state.pdf_chunks = chunks
                    st.session_state.pdf_index = index
                    st.session_state.embed_model = embed_model
                    kpi("Chunks", f"{len(chunks):,}")
            except Exception as e:
                st.error(f"PDF read error: {e}")

    with c2:
        st.markdown("### 🤖 Ask the PDF")
        user_q = st.text_input("Type a focused question", "What are the top risks mentioned?")
        ask = st.button("Generate Answer", type="primary", use_container_width=True)
        if ask:
            if not API_KEY:
                st.error("Missing OPENAI_API_KEY (Secrets or .env).")
            elif not FAISS_OK or "pdf_index" not in st.session_state:
                st.error("Please upload & index a PDF first (and ensure FAISS is available).")
            else:
                try:
                    hits = top_k_chunks(
                        user_q,
                        st.session_state.pdf_chunks,
                        st.session_state.pdf_index,
                        st.session_state.embed_model,
                        k=6,
                    )
                    context = "\n\n".join(hits)
                    prompt = f"""
You are an equity research analyst. Using ONLY the provided context from a company's report,
answer the user's question in a clear, structured summary with bullet points.
Avoid quoting raw text or showing long passages. If context is insufficient, state what more is needed.

Question: {user_q}

Context:
{context}
"""
                    with st.spinner("Synthesizing answer…"):
                        resp = client.chat.completions.create(
                            model="meta-llama/llama-3.1-8b-instruct",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                    st.markdown("#### ✅ Answer")
                    st.write(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"Answering error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Tab 2: Screener (with logos)
# =========================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown("### 📊 Daily Top-5 Screener (Enhanced Factors + Logos)")
    st.caption("Enter NSE tickers without `.NS` suffix, e.g., `TCS, INFY, RELIANCE, HDFCBANK, ITC`")
    symbols = st.text_input(
        "Enter NSE stock symbols (comma-separated):",
        "TCS, INFY, RELIANCE, HDFCBANK, ITC, LT, SBIN, ICICIBANK, ULTRACEMCO"
    )
    run = st.button("Run Screener", type="primary")

    if run:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        with st.spinner("Fetching fundamentals, prices & logos…"):
            df = get_stock_snapshot(syms)

        if df.empty:
            st.error("Could not fetch any data. Try different symbols.")
        else:
            picks = df.sort_values("Score", ascending=False).head(5).reset_index(drop=True)

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Universe", len(df))
            k2.metric("Top 5 Mean Score", f"{picks['Score'].mean():.2f}")
            k3.metric("Avg PE (Top 5)", f"{picks['PE'].replace([np.inf, -np.inf], np.nan).fillna(0).mean():.2f}")
            k4.metric("Avg Momentum 30d", f"{picks['Momentum30d'].mean():.1f}%")

            st.markdown("#### 🏆 Top 5 Picks (with logos)")

            # 2-column grid of rich cards
            cols = st.columns(2)
            smin, smax = float(picks["Score"].min()), float(picks["Score"].max())

            for i, row in picks.iterrows():
                with cols[i % 2]:
                    # Logo (fallback placeholder if missing)
                    logo_html = (
                        f"<img src='{row['Logo']}' width='40' style='border-radius:8px;border:1px solid rgba(255,255,255,0.12);'/>"
                        if row['Logo'] else
                        "<div style='width:40px;height:40px;border-radius:8px;border:1px solid rgba(255,255,255,0.12);display:flex;align-items:center;justify-content:center;opacity:.8;'>📈</div>"
                    )

                    score = float(row["Score"])
                    pct = 100 if smax == smin else int(100 * (score - smin) / (smax - smin))

                    mom = float(row["Momentum30d"])
                    mom_cls = "good" if mom >= 0 else "bad"
                    pe_txt = f"{row['PE']:.2f}" if np.isfinite(row["PE"]) else "—"
                    last_txt = f"₹{row['LastPrice']:.2f}" if row.get("LastPrice") else "—"
                    name = row.get("ShortName") or row["Symbol"]

                    st.markdown(f"""
                    <div class="pick-card" style="margin:8px 0;">
                      {logo_html}
                      <div class="meta" style="flex:1;">
                        <div class="pick-title">{name} <span style="opacity:.6;">({row['Symbol']})</span></div>
                        <div class="pick-sub">Price: <b>{last_txt}</b> • PE: <b>{pe_txt}</b> • Momentum 30d: <b class="{mom_cls}">{mom:+.1f}%</b></div>
                        <div class="pick-sub">Score</div>
                        <div class="scorebar"><div style="width:{pct}%"></div></div>
                      </div>
                      <div class="kpill">Score&nbsp;{score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Detailed table below
            st.markdown("#### Details")
            st.dataframe(
                style_screener_table(picks[[
                    "Symbol","Score","PE","PB","EVEBITDA","ROE","ProfitMargin","OperatingMargin",
                    "RevenueGrowth","EarningsGrowth","DebtToEquity","Momentum30d"
                ]]),
                use_container_width=True
            )

            # LLM rationale (optional)
            if API_KEY:
                rationale_prompt = f"""
Summarize in 4-7 bullets why these Indian stocks might rank as top picks today,
given value (PE/PB/EV/EBITDA), quality (ROE/margins), growth (revenue & earnings),
risk (debt/equity), and momentum (30d). Be concise and non-promissory.

Data:
{picks.to_string(index=False)}
"""
                try:
                    with st.spinner("Generating brief rationale…"):
                        rr = client.chat.completions.create(
                            model="meta-llama/llama-3.1-8b-instruct",
                            messages=[{"role":"user","content": rationale_prompt}],
                            temperature=0.3,
                        )
                    st.markdown("#### 📝 Brief Rationale")
                    st.write(rr.choices[0].message.content)
                except Exception:
                    st.caption("Rationale skipped (API unavailable).")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Tab 3: Market Wrap
# =========================
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown("### 📰 Daily Market Wrap (NIFTY 50 / SENSEX)")
    wrap_df, ts = get_market_wrap()
    if wrap_df.empty:
        st.error("Could not fetch index data.")
    else:
        c1, c2, c3 = st.columns(3)
        last_nifty = wrap_df.loc[wrap_df["Index"] == "NIFTY 50", "Last"].values[0] if "NIFTY 50" in wrap_df["Index"].values else None
        last_sensex = wrap_df.loc[wrap_df["Index"] == "SENSEX", "Last"].values[0] if "SENSEX" in wrap_df["Index"].values else None
        # KPIs
        def render_kpi(label, value, delta=None):
            if delta is None:
                kpi(label, value)
            else:
                st.metric(label, value, delta)

        render_kpi("Last Updated", ts)
        if last_nifty is not None:
            c1.metric("NIFTY 50 (Last)", f"{last_nifty:,.0f}",
                      f"{wrap_df.loc[wrap_df['Index']=='NIFTY 50','DayChangePct'].values[0]:+.2f}%")
        if last_sensex is not None:
            c2.metric("SENSEX (Last)", f"{last_sensex:,.0f}",
                      f"{wrap_df.loc[wrap_df['Index']=='SENSEX','DayChangePct'].values[0]:+.2f}%")
        c3.metric("1-Month View", "See table ↓")

        st.markdown("#### Index Snapshot")
        fmt = wrap_df.copy()
        fmt["Last"] = fmt["Last"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        fmt["DayChangePct"] = fmt["DayChangePct"].map(lambda x: f"{x:+.2f}%")
        fmt["MonthChangePct"] = fmt["MonthChangePct"].map(lambda x: f"{x:+.2f}%")
        st.dataframe(fmt, use_container_width=True)

        # Short LLM wrap
        if API_KEY:
            try:
                wrap_prompt = f"""
Write a short market wrap (4-6 bullet points) for Indian markets based on this index snapshot.
Mention NIFTY 50 and SENSEX direction (up/down) with approximate % moves (rounded),
and add one neutral closing line. Avoid advice/guarantees.

Data:
{wrap_df.to_string(index=False)}
"""
                with st.spinner("Summarizing market wrap…"):
                    wrap_resp = client.chat.completions.create(
                        model="meta-llama/llama-3.1-8b-instruct",
                        messages=[{"role":"user","content": wrap_prompt}],
                        temperature=0.3,
                    )
                st.markdown("#### ✍️ Summary")
                st.write(wrap_resp.choices[0].message.content)
            except Exception:
                st.caption("Summary skipped (API unavailable).")

    st.markdown('</div>', unsafe_allow_html=True)
