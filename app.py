import os
import json
import re
from collections import Counter
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ============================================================
# NLP PIPELINE
# ============================================================

def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    
    # Overall sentiment
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    
    # Classify sentiment
    if polarity > 0.3:
        sentiment_label = "Positive"
        sentiment_color = "#22c55e"
        sentiment_emoji = "😊"
    elif polarity > 0.05:
        sentiment_label = "Slightly Positive"
        sentiment_color = "#86efac"
        sentiment_emoji = "🙂"
    elif polarity > -0.05:
        sentiment_label = "Neutral"
        sentiment_color = "#fbbf24"
        sentiment_emoji = "😐"
    elif polarity > -0.3:
        sentiment_label = "Slightly Negative"
        sentiment_color = "#fb923c"
        sentiment_emoji = "😕"
    else:
        sentiment_label = "Negative"
        sentiment_color = "#ef4444"
        sentiment_emoji = "😞"
    
    # Subjectivity classification
    if subjectivity > 0.6:
        subjectivity_label = "Highly Subjective (Opinion-based)"
    elif subjectivity > 0.4:
        subjectivity_label = "Moderately Subjective"
    else:
        subjectivity_label = "Mostly Objective (Fact-based)"
    
    # Sentence-level sentiment breakdown
    sentences = sent_tokenize(text)
    sentence_sentiments = []
    for sent in sentences[:10]:  # Limit to first 10 sentences
        sent_blob = TextBlob(sent)
        sent_pol = round(sent_blob.sentiment.polarity, 3)
        if sent_pol > 0.05:
            s_label = "Positive"
            s_color = "#22c55e"
        elif sent_pol < -0.05:
            s_label = "Negative"
            s_color = "#ef4444"
        else:
            s_label = "Neutral"
            s_color = "#fbbf24"
        sentence_sentiments.append({
            "text": sent.strip(),
            "polarity": sent_pol,
            "label": s_label,
            "color": s_color
        })
    
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment_label": sentiment_label,
        "sentiment_color": sentiment_color,
        "sentiment_emoji": sentiment_emoji,
        "subjectivity_label": subjectivity_label,
        "sentence_sentiments": sentence_sentiments,
        "polarity_percentage": round((polarity + 1) / 2 * 100, 1)
    }


def extract_topics(text):
    """Extract key topics using frequency analysis and NLP."""
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Add custom stop words for feedback context
    custom_stops = {
        'would', 'could', 'also', 'really', 'much', 'like', 'get',
        'got', 'one', 'use', 'used', 'using', 'make', 'made',
        'need', 'want', 'think', 'know', 'thing', 'things', 'even',
        'still', 'app', 'application', 'product', 'service', 'software',
        'please', 'thanks', 'thank', 'well', 'good', 'bad', 'great',
        'nice', 'better', 'best', 'worst', 'every', 'always', 'never',
        'time', 'way', 'lot', 'many', 'able', 'since', 'back',
        'however', 'though', 'although', 'overall', 'especially',
        'something', 'anything', 'everything', 'nothing', 'sometimes'
    }
    all_stops = stop_words.union(custom_stops)
    
    # Filter tokens: keep only alphabetic tokens, length > 2, not stopwords
    filtered = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in all_stops]
    
    # Single word frequency
    word_freq = Counter(filtered)
    
    # Bigram extraction for compound topics
    bigram_list = list(ngrams(filtered, 2))
    bigram_freq = Counter([' '.join(bg) for bg in bigram_list])
    
    # Get top topics
    top_words = word_freq.most_common(8)
    top_bigrams = bigram_freq.most_common(5)
    
    # Combine and deduplicate
    topics = []
    seen_words = set()
    
    # Add bigrams first (more informative)
    for bigram, count in top_bigrams:
        if count >= 2:
            topics.append({"topic": bigram, "count": count, "type": "phrase"})
            for w in bigram.split():
                seen_words.add(w)
    
    # Add single words not already covered by bigrams
    for word, count in top_words:
        if word not in seen_words and count >= 2:
            topics.append({"topic": word, "count": count, "type": "keyword"})
    
    # If we have fewer than 3 topics, add top words regardless of count
    if len(topics) < 3:
        for word, count in top_words:
            if word not in seen_words:
                topics.append({"topic": word, "count": count, "type": "keyword"})
                seen_words.add(word)
                if len(topics) >= 5:
                    break
    
    return topics[:8]


def extract_key_phrases(text):
    """Extract noun phrases as key themes."""
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    
    # Count occurrences
    np_freq = Counter(noun_phrases)
    top_phrases = np_freq.most_common(6)
    
    return [{"phrase": phrase, "count": count} for phrase, count in top_phrases if len(phrase) > 2]


def generate_rule_based_recommendations(sentiment_data, topics, key_phrases):
    """Generate product recommendations based on analysis results."""
    recommendations = []
    polarity = sentiment_data["polarity"]
    subjectivity = sentiment_data["subjectivity"]
    sentence_sentiments = sentiment_data["sentence_sentiments"]
    
    # Count negative sentences
    neg_sentences = [s for s in sentence_sentiments if s["label"] == "Negative"]
    pos_sentences = [s for s in sentence_sentiments if s["label"] == "Positive"]
    
    # Overall sentiment-based recommendations
    if polarity < -0.2:
        recommendations.append({
            "priority": "High",
            "priority_color": "#ef4444",
            "title": "Critical: Address Negative Customer Sentiment",
            "description": "The feedback shows significantly negative sentiment. Immediate investigation is recommended to identify root causes and prevent churn.",
            "actions": [
                "Conduct targeted user research with dissatisfied customers",
                "Set up an urgent review meeting with the product team",
                "Consider reaching out to the feedback author for deeper understanding"
            ]
        })
    elif polarity < 0:
        recommendations.append({
            "priority": "Medium",
            "priority_color": "#f59e0b",
            "title": "Monitor: Mildly Negative Sentiment Detected",
            "description": "The feedback indicates mild dissatisfaction. Proactive monitoring and incremental improvements are advisable.",
            "actions": [
                "Add this feedback to the product improvement backlog",
                "Track whether similar sentiment appears in other feedback sources",
                "Schedule a review in the next sprint planning session"
            ]
        })
    elif polarity > 0.3:
        recommendations.append({
            "priority": "Low",
            "priority_color": "#22c55e",
            "title": "Leverage: Strongly Positive Feedback",
            "description": "This feedback reflects high satisfaction. Consider leveraging this for testimonials and identifying what is working well to replicate.",
            "actions": [
                "Identify the specific features or experiences driving satisfaction",
                "Consider requesting a testimonial or case study from this user",
                "Document successful patterns for replication across the product"
            ]
        })
    
    # Topic-based recommendations
    topic_names = [t["topic"] for t in topics]
    topic_text = " ".join(topic_names)
    
    # Performance-related topics
    perf_keywords = ['slow', 'loading', 'speed', 'performance', 'lag', 'crash', 'crashes', 'bug', 'error', 'freeze']
    if any(kw in topic_text for kw in perf_keywords):
        recommendations.append({
            "priority": "High",
            "priority_color": "#ef4444",
            "title": "Performance Issues Identified",
            "description": "Customer feedback mentions performance-related concerns. Technical investigation is recommended.",
            "actions": [
                "Run performance profiling on the identified areas",
                "Review recent deployments for potential regression",
                "Prioritize performance fixes in the current or next sprint",
                "Set up performance monitoring alerts if not already in place"
            ]
        })
    
    # UI/UX-related topics
    ux_keywords = ['interface', 'design', 'layout', 'navigation', 'confusing', 'intuitive', 'ui', 'ux', 'difficult', 'complicated', 'user experience']
    if any(kw in topic_text for kw in ux_keywords):
        recommendations.append({
            "priority": "Medium",
            "priority_color": "#f59e0b",
            "title": "User Experience Improvement Opportunity",
            "description": "Feedback suggests user experience concerns. A UX review may be beneficial.",
            "actions": [
                "Conduct a usability audit of the mentioned features",
                "Review user flow analytics to identify drop-off points",
                "Consider A/B testing alternative interface designs",
                "Gather additional feedback through user interviews"
            ]
        })
    
    # Feature request-related topics
    feature_keywords = ['feature', 'add', 'missing', 'wish', 'option', 'ability', 'functionality', 'support', 'integration', 'export', 'import']
    if any(kw in topic_text for kw in feature_keywords):
        recommendations.append({
            "priority": "Medium",
            "priority_color": "#f59e0b",
            "title": "Feature Enhancement Requests Detected",
            "description": "The feedback contains feature requests or mentions of missing functionality.",
            "actions": [
                "Add identified feature requests to the product backlog",
                "Evaluate demand by checking if similar requests exist from other users",
                "Score using a prioritization framework (RICE, MoSCoW, or AI-driven)",
                "Communicate the roadmap status to the requesting users"
            ]
        })
    
    # Pricing-related topics
    price_keywords = ['price', 'pricing', 'expensive', 'cost', 'cheap', 'affordable', 'subscription', 'plan', 'billing', 'payment', 'value']
    if any(kw in topic_text for kw in price_keywords):
        recommendations.append({
            "priority": "Medium",
            "priority_color": "#f59e0b",
            "title": "Pricing and Value Perception Feedback",
            "description": "Feedback mentions pricing or value-related concerns.",
            "actions": [
                "Review competitive pricing analysis",
                "Evaluate whether feature gating aligns with perceived value",
                "Consider conducting a willingness-to-pay survey",
                "Assess if a different pricing tier could address the concern"
            ]
        })
    
    # Support-related topics
    support_keywords = ['support', 'help', 'response', 'customer service', 'ticket', 'wait', 'resolve', 'resolution']
    if any(kw in topic_text for kw in support_keywords):
        recommendations.append({
            "priority": "Medium",
            "priority_color": "#f59e0b",
            "title": "Customer Support Experience Feedback",
            "description": "The feedback references customer support experiences.",
            "actions": [
                "Review support response time metrics",
                "Identify common support queries for self-service documentation",
                "Consider implementing AI-powered chatbot for initial triage",
                "Train support team on the specific issues mentioned"
            ]
        })
    
    # Subjectivity-based recommendation
    if subjectivity > 0.6:
        recommendations.append({
            "priority": "Low",
            "priority_color": "#3b82f6",
            "title": "Feedback is Highly Opinion-Based",
            "description": "This feedback is predominantly subjective. While valuable for understanding user sentiment, consider supplementing with quantitative data before making product decisions.",
            "actions": [
                "Cross-reference with product analytics data for objective validation",
                "Conduct a broader survey to check if the sentiment is widespread",
                "Use this as a qualitative input alongside quantitative metrics"
            ]
        })
    
    # Mixed sentiment recommendation
    if len(neg_sentences) > 0 and len(pos_sentences) > 0:
        recommendations.append({
            "priority": "Low",
            "priority_color": "#3b82f6",
            "title": "Mixed Sentiment: Identify Strengths and Weaknesses",
            "description": f"The feedback contains both positive ({len(pos_sentences)} sentences) and negative ({len(neg_sentences)} sentences) elements. This suggests partial satisfaction with room for improvement.",
            "actions": [
                "Separate the positive elements (what to preserve) from negative elements (what to fix)",
                "Ensure product changes do not compromise the positively received aspects",
                "Use the positive elements as differentiators in competitive positioning"
            ]
        })
    
    # General recommendation if few specific ones
    if len(recommendations) < 2:
        recommendations.append({
            "priority": "Low",
            "priority_color": "#3b82f6",
            "title": "General: Log and Monitor",
            "description": "No urgent action items detected. Log this feedback for pattern analysis over time.",
            "actions": [
                "Add to the feedback repository for longitudinal analysis",
                "Tag with relevant product area for future reference",
                "Review in aggregate during the next product review cycle"
            ]
        })
    
    return recommendations


def generate_llm_recommendations(text, sentiment_data, topics, key_phrases):
    """Generate recommendations using Anthropic Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    
    try:
        import requests
        
        topic_str = ", ".join([t["topic"] for t in topics])
        phrase_str = ", ".join([p["phrase"] for p in key_phrases]) if key_phrases else "None identified"
        
        prompt = f"""You are a senior product management advisor. Analyze the following customer feedback analysis results and provide actionable product recommendations.

CUSTOMER FEEDBACK:
"{text[:1500]}"

ANALYSIS RESULTS:
- Sentiment: {sentiment_data['sentiment_label']} (Polarity: {sentiment_data['polarity']}, Subjectivity: {sentiment_data['subjectivity']})
- Key Topics: {topic_str}
- Key Phrases: {phrase_str}

Based on this analysis, provide exactly 4 actionable product recommendations. For each recommendation, specify:
1. Priority (High/Medium/Low)
2. A short title (max 8 words)
3. A one-sentence description
4. 2-3 specific action items

Respond ONLY in this exact JSON format, no other text:
[
  {{
    "priority": "High",
    "title": "...",
    "description": "...",
    "actions": ["...", "...", "..."]
  }}
]"""
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["content"][0]["text"]
            # Clean and parse JSON
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```json?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            
            recs = json.loads(content)
            
            # Format recommendations
            formatted = []
            for rec in recs:
                priority = rec.get("priority", "Medium")
                color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
                formatted.append({
                    "priority": priority,
                    "priority_color": color_map.get(priority, "#3b82f6"),
                    "title": rec.get("title", "Recommendation"),
                    "description": rec.get("description", ""),
                    "actions": rec.get("actions", [])
                })
            return formatted
    except Exception as e:
        print(f"LLM recommendation error: {e}")
    
    return None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("feedback", "").strip()
    
    if not text:
        return jsonify({"error": "Please provide feedback text to analyze."}), 400
    
    if len(text) < 20:
        return jsonify({"error": "Please provide at least a few sentences for meaningful analysis."}), 400
    
    # Run NLP pipeline
    sentiment = analyze_sentiment(text)
    topics = extract_topics(text)
    key_phrases = extract_key_phrases(text)
    
    # Try LLM recommendations first, fallback to rule-based
    recommendations = generate_llm_recommendations(text, sentiment, topics, key_phrases)
    recommendation_source = "AI-Generated (Claude LLM)"
    
    if recommendations is None:
        recommendations = generate_rule_based_recommendations(sentiment, topics, key_phrases)
        recommendation_source = "Rule-Based NLP Analysis"
    
    # Word and sentence stats
    word_count = len(text.split())
    sentence_count = len(sent_tokenize(text))
    
    return jsonify({
        "sentiment": sentiment,
        "topics": topics,
        "key_phrases": key_phrases,
        "recommendations": recommendations,
        "recommendation_source": recommendation_source,
        "stats": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "topics_found": len(topics),
            "phrases_found": len(key_phrases)
        }
    })


@app.route("/sample")
def get_sample():
    """Return sample feedback for demo purposes."""
    samples = [
        {
            "title": "SaaS Dashboard - Mixed Feedback",
            "text": "The analytics dashboard has great visualizations and the real-time data updates are impressive. However, the loading time is extremely frustrating, especially when switching between different date ranges. The export feature crashes about half the time when trying to download CSV files. I also wish there was better integration with our CRM system. The mobile experience is terrible - the charts are completely unreadable on my phone. Customer support has been helpful when I reach out, but the response time could be much faster. Overall, the core product is solid but these quality issues are making us reconsider our subscription."
        },
        {
            "title": "E-commerce Platform - Negative Feedback",
            "text": "The checkout process is painfully slow and confusing. I have been trying to complete my purchase for the last 30 minutes. The payment page keeps refreshing and losing my cart items. The search functionality is broken - I searched for running shoes and got results for kitchen appliances. The filters do not work properly either. When I try to sort by price, nothing changes. I contacted customer support through chat but waited 45 minutes with no response. The website design looks outdated compared to competitors. I used to love shopping here but the experience has deteriorated significantly over the past few months. The mobile app is even worse, constantly crashing and logging me out. I am seriously considering switching to a competitor."
        },
        {
            "title": "Project Management Tool - Positive Feedback",
            "text": "We switched to this project management tool six months ago and it has transformed how our team operates. The Kanban board is intuitive and the drag-and-drop functionality works flawlessly. The automated workflow features save us hours every week. Integration with Slack and Google Drive was seamless and took less than five minutes to set up. The reporting dashboard gives our leadership team exactly the visibility they need. The recent AI-powered task suggestions feature is surprisingly accurate and helps with sprint planning. The pricing is reasonable for the value we get. Our team productivity has increased noticeably since adopting this tool. The only minor complaint is that the notification settings could be more granular, but that is a small issue compared to the overall value."
        },
        {
            "title": "Marketing Automation Platform - Feature Requests",
            "text": "The email campaign builder is decent but lacks advanced personalization options. We need the ability to create dynamic content blocks based on user behavior segments. The A/B testing feature only allows two variants which is limiting for our needs. We would love to see multivariate testing capabilities. The analytics are basic compared to what competitors offer. There is no cohort analysis, no funnel visualization, and the attribution modeling is overly simplistic. Integration with our data warehouse would be extremely valuable. We also need better API documentation. The current docs are outdated and have missing endpoints. The workflow automation is promising but needs conditional branching logic and better error handling. If these features were added, we would happily upgrade to the enterprise tier."
        }
    ]
    return jsonify(samples)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
