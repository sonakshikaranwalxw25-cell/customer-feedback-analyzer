# AI Customer Feedback Analyzer

**MRP-1 Project | IIM Ranchi | EMBA 2025-27**

An AI-powered Customer Feedback Analysis System that leverages Natural Language Processing (NLP) to perform automated sentiment analysis, topic extraction, and generate actionable product recommendations.

## Features

- **Sentiment Analysis** — Polarity and subjectivity scoring using TextBlob NLP
- **Topic Extraction** — Automated keyword and bigram frequency analysis using NLTK
- **Key Phrase Detection** — Noun phrase extraction for thematic understanding
- **Sentence-Level Breakdown** — Individual sentiment scoring per sentence
- **AI Recommendations** — LLM-powered product recommendations (Claude API) with rule-based fallback
- **Sample Datasets** — Pre-loaded feedback scenarios for demonstration

## Tech Stack

- **Backend:** Python, Flask
- **NLP:** TextBlob, NLTK
- **AI/LLM:** Anthropic Claude API
- **Deployment:** Render (Gunicorn)

## Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd feedback-analyzer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet')"

# (Optional) Set Anthropic API key for LLM recommendations
export ANTHROPIC_API_KEY=your_api_key_here

# Run the application
python app.py
```

Visit `http://localhost:5000` in your browser.

## Deploying to Render

1. Push this repository to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repo
4. Set Build Command: `chmod +x build.sh && ./build.sh`
5. Set Start Command: `gunicorn app:app`
6. (Optional) Add environment variable `ANTHROPIC_API_KEY` for LLM recommendations

## Project Details

- **Student:** Sonakshi Karanwal (XW035-25)
- **Guide:** Prof. Shibashish Chakraborty
- **Course:** Major Research Project - 1
- **Institute:** Indian Institute of Management, Ranchi
