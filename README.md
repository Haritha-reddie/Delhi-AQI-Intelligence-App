# 🌫️ Delhi AQI Intelligence App

> An end-to-end AI-powered air quality analytics platform built with Python, Streamlit, Prophet, and Claude AI, featuring real-time dashboards, ML forecasting, and natural language data querying.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-ML_Forecasting-orange)
![Claude AI](https://img.shields.io/badge/Claude_AI-Anthropic-blueviolet)
![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-yellow?logo=googlecolab)

---

## 📌 Project Overview

This project transforms raw Delhi air quality sensor data (2025–2026) into a fully interactive intelligence platform. It goes beyond basic charts, users can explore pollution trends, forecast future AQI levels using machine learning, and ask natural language questions to an AI analyst powered by Claude.

**Built as a portfolio project demonstrating Full-Stack AI Engineering skills across the entire data-to-insight pipeline.**

---

## 🎯 Problem Statement

Delhi consistently ranks among the world's most polluted cities. Raw air quality data from monitoring stations is collected continuously but remains difficult to interpret for the general public, policymakers, and researchers. This project solves that by:

- Making complex pollution data **visually intuitive**
- Using **ML forecasting** to predict future AQI levels
- Enabling **conversational AI** to answer questions about the data in plain English

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Delhi Air Quality Time Series 2025–2026](https://www.kaggle.com/datasets/jaspreetsingh9652/delhi-air-quality-time-series-data-2025-to-2026-05) |
| **Date Range** | February 2025 – May 2026 |
| **Format** | Long format CSV (one row per parameter reading) |
| **Stations** | Anand Vihar, Punjabi Bagh, Pusa, R K Puram |
| **Parameters** | AQI, PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3, Temperature, Humidity, Wind |
| **Records** | ~120,000+ readings |

### Raw Data Structure
```
DateTime            | Locations  | Parameters | Values | Units | hour
2025-02-18 20:00:00 | R K Puram  | no         | 93.2   | ppb   | 20
2025-02-18 20:15:00 | R K Puram  | pm25       | 87.4   | µg/m³ | 20
```

### After Processing (Pivoted Wide Format)
```
datetime            | location   | aqi  | pm25  | pm10  | no   | no2  | co   | ...
2025-02-18 20:00:00 | R K Puram  | 97.0 | 87.4  | 213.0 | 93.2 | 45.1 | 1.2  | ...
```

---

## 🏗️ Architecture

```
delhi-aqi-app/
│
├── backend/
│   ├── data_loader.py       ← ETL pipeline: load, clean, pivot, detect columns
│   ├── forecast.py          ← Prophet ML forecasting + fallback rolling average
│   └── main.py              ← FastAPI REST API (optional)
│
├── frontend/
│   └── app.py               ← Streamlit dashboard (4 tabs)
│
├── ai/
│   └── agent.py             ← Claude AI chat analyst
│
├── data/
│   └── delhi_aqi.csv        ←  Kaggle CSV goes here
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ✨ Features

### 📊 Tab 1 — Interactive Dashboard
- **AQI Trend Chart** — Time-series line chart across all selected stations
- **AQI Category Donut Chart** — Distribution across Good / Satisfactory / Moderate / Poor / Very Poor / Severe
- **Station Box Plot** — Spread and outliers of AQI per monitoring station
- **Pollutant Correlation Heatmap** — Pearson correlation matrix across all pollutants (PM2.5, NO2, CO, O3, etc.)

### 🗺️ Tab 2 — Location Heatmap
- Monthly AQI averages by station — instantly reveals seasonal pollution patterns
- Winter months (Nov–Jan) visually spike red, summer months show lower pollution
- White cells = missing data for that station/month

### 🔮 Tab 3 — ML Forecasting
- Powered by **Facebook Prophet** with yearly + weekly seasonality
- Forecasts AQI 7–60 days into the future
- Shows **confidence intervals** (shaded band)
- Falls back to rolling average if Prophet is unavailable
- Verdict card shows predicted AQI category with color coding

### 🤖 Tab 4 — AI Chat Analyst
- Powered by **Anthropic Claude** (`claude-sonnet-4-6`)
- Context-aware: Claude receives real stats from your filtered dataset
- ChatGPT-style interface: history on top, input at bottom
- Quick question chips for one-click queries
- Multi-turn conversation with memory
- Example answers include tables, bullet points, health recommendations

---

## 🔬 Technical Deep Dive

### Data Pipeline (`data_loader.py`)

The CSV arrives in **long format** (one row per pollutant reading). The pipeline:

1. **Normalizes column names** — strips whitespace, lowercases
2. **Parses datetime** — auto-detects date/time columns
3. **Pivots to wide format** — `pivot_table(index=['datetime','location'], columns='Parameters', values='Values')`
4. **Auto-detects AQI column** — checks for `aqi`, falls back to `pm25`
5. **Cleans outliers** — drops rows where all numeric values are NaN

```python
pivot = df.pivot_table(
    index=['datetime', 'Locations'],
    columns='Parameters',
    values='Values',
    aggfunc='mean'
).reset_index()
```

### Forecasting (`forecast.py`)

Uses **Facebook Prophet** for robust time-series forecasting:

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # controls trend flexibility
)
model.fit(daily_aqi_df)
future = model.make_future_dataframe(periods=30, freq='D')
forecast = model.predict(future)
```

Key decisions:
- Resamples to **daily mean** before training (reduces noise)
- `changepoint_prior_scale=0.05` prevents overfitting to spikes
- Returns Plotly figure with actual vs forecast + confidence band

### AI Agent (`agent.py`)

The agent builds a **compact data context** from the filtered DataFrame and injects it into Claude's prompt:

```python
context = f"""
Dataset: {date_range}
Locations: {locations}
Stats:
  AQI: mean=101.4, max=970, min=12, latest=85
  PM25: mean=87.2, max=450, min=8, latest=71
  NO2: mean=39.6, max=180, min=2, latest=59
"""
```

Claude then answers with this real data as grounding — not hallucinated values.

**System prompt instructs Claude to:**
- Always explain AQI numbers in plain English
- Give health advice when relevant
- Use India's AQI scale (0–50 Good → 400+ Severe)
- Be data-driven but conversational

### AQI Scale (India Standard)

| Range | Category | Color |
|---|---|---|
| 0–50 | Good | 🟢 Green |
| 51–100 | Satisfactory | 🟡 Yellow |
| 101–200 | Moderate | 🟠 Orange |
| 201–300 | Poor | 🔴 Red |
| 301–400 | Very Poor | 🟣 Purple |
| 400+ | Severe | 🟤 Maroon |

---

## 📸 Screenshots

### Dashboard — Trend & Charts
![Dashboard](screenshots/dashboard.png)

### Heatmap — Monthly AQI by Station
![Heatmap](screenshots/heatmap.png)

### Forecast — Prophet ML Prediction
![Forecast](screenshots/forecast.png)

### AI Chat — Claude Analyst
![AI Chat](screenshots/ai_chat.png)

---

## 🚀 How to Run

### Option A — Google Colab (Recommended, No Setup)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Delhi_AQI_App.ipynb`
3. Run all cells top to bottom
4. Upload your CSV when prompted
5. Add your Anthropic API key in Cell 4
6. Click the Cloudflare tunnel URL printed by Cell 9

### Option B — Local (Cursor / VS Code)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/delhi-aqi-app
cd delhi-aqi-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Place your CSV at: data/delhi_aqi.csv

# 4. Set API key
cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 5. Run
streamlit run frontend/app.py
```

Open **http://localhost:8501**

### Option C — Docker

```bash
docker build -t delhi-aqi-app .
docker run -p 8501:8501 --env-file .env delhi-aqi-app
```

---

## ⚙️ Requirements

```
streamlit==1.35.0
anthropic>=0.40.0
plotly==5.22.0
prophet==1.1.5
pandas==2.2.2
numpy==1.26.4
python-dotenv==1.0.1
fastapi==0.111.0
uvicorn==0.30.1
```

Get a free Anthropic API key at: https://console.anthropic.com

---

## 🧠 Skills Demonstrated

| Skill | How |
|---|---|
| **Data Engineering** | ETL pipeline, long→wide pivot, auto column detection |
| **Machine Learning** | Time-series forecasting with Prophet, seasonality modeling |
| **AI / LLM Integration** | Claude API, prompt engineering, context injection, multi-turn chat |
| **Full-Stack Development** | Streamlit UI, FastAPI backend, REST endpoints |
| **Data Visualization** | Plotly charts, heatmaps, correlation matrices |
| **DevOps** | Docker, Cloudflare tunneling, Google Colab deployment |
| **Python** | Pandas, NumPy, subprocess management, environment config |

---

## 💡 Key Insights from the Data

Based on analysis of the 2025–2026 dataset:

- **Anand Vihar** consistently shows the highest AQI — driven by heavy traffic from the Inter-State Bus Terminal
- **Winter months (Nov–Jan)** show the worst air quality — AQI spikes to 300–970 (Severe)
- **Summer months (Apr–Jun)** show relatively better air quality — AQI typically 50–150
- **PM2.5 and PM10 are strongly correlated (r=0.83)** — both driven by vehicle emissions and construction dust
- **Temperature negatively correlates with AQI (r=-0.56)** — higher temperatures disperse pollutants
- **O3 (ozone) negatively correlates with NO (r=-0.42)** — classic photochemical relationship

---

## 🗺️ Future Improvements

- [ ] Add real-time data ingestion via CPCB API
- [ ] Health risk calculator (age + AQI → personal risk score)
- [ ] Alert system — email/SMS when AQI exceeds threshold
- [ ] Multi-city comparison (Mumbai, Bangalore, Chennai)
- [ ] Map visualization with station GPS coordinates
- [ ] Export reports as PDF

---

## 👤 Author

**Haritha**
- Built with Python, Streamlit, Prophet, Claude AI
- Dataset: Delhi Air Quality 2025–2026 via Kaggle
- Deployed on Google Colab + Cloudflare Tunnel

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- Dataset by [Jaspreet Singh](https://www.kaggle.com/jaspreetsingh9652) on Kaggle
- Forecasting by [Facebook Prophet](https://facebook.github.io/prophet/)
- AI powered by [Anthropic Claude](https://anthropic.com)
- Dashboard built with [Streamlit](https://streamlit.io)
