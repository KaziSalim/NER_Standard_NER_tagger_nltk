import os
os.environ['JAVAHOME'] = 'D:/Data Scientist/Unsupervised Learning/Text Mining & NLP/GitHub/NER/jdk-17.0.9'  # Replace with your actual path


import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Set the path to the Stanford NER JAR file and model file
stanford_ner_path = 'D:/Data Scientist/Unsupervised Learning/Text Mining & NLP/GitHub/NER/stanford-ner-2020-11-17'
jar_path = f'{stanford_ner_path}/stanford-ner.jar'
model_path = f'{stanford_ner_path}/classifiers/english.all.3class.distsim.crf.ser.gz'


# Initialize the Stanford NER tagger
ner_tagger = StanfordNERTagger(model_path, jar_path, encoding='utf-8')

def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)
    print("Tokens:", tokens)  

    # Removing stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]

    return " ".join(tokens)

def perform_ner(text):
    # Tokenize the sentence
    tokens = word_tokenize(text)

    # Perform NER using the Stanford NER tagger
    ner_results = ner_tagger.tag(tokens)

    # Print the named entities
    print("Named Entities:")
    for token, tag in ner_results:
        if tag != 'O':
            print(f"{token} - {tag}")

# Example sentence
raw_text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976."

# Preprocess the text
preprocessed_text = preprocess_text(raw_text)

# Perform NER on preprocessed text
perform_ner(preprocessed_text)
