#!/bin/bash
pip install -r requirements.txt
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('brown')
"
python -c "import textblob; textblob.download_corpora.download_all()"
