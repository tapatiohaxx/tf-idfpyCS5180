#-------------------------------------------------------------------------
# AUTHOR: Matthew Plascencia
# FILENAME: Python-tf-idf.py
# SPECIFICATION: A python program to do the work for problem 7:It foes the stopword removal (removes pronouns/conjugations/etc)
# and then does stemming. After that it indexes the words and does TF, IDF, and TF-IDF. It prints them in easy to read tables and matrices.
# FOR: CS 5180- Assignment #1
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

import csv
import re
import math
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

documents = []
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            documents.append(row[0])

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def preprocess(text):
    words = tokenize(text)
    words_filtered = [ps.stem(word) for word in words if word not in stop_words]
    return words_filtered

terms = set()
processed_docs = []
for doc in documents:
    processed = preprocess(doc)
    processed_docs.append(processed)
    terms.update(processed)

terms = sorted(terms)

def compute_tf(word_dict, doc):
    tf_dict = {}
    doc_length = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_length)
    return tf_dict

def compute_idf(doc_list, vocab):
    N = len(doc_list)
    idf_dict = dict.fromkeys(vocab, 0)
    for doc in doc_list:
        for word in vocab:
            if doc.count(word) > 0:
                idf_dict[word] += 1
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val)) if val > 0 else 0
    return idf_dict

def compute_tf_idf(tf_doc, idfs):
    tf_idf = {}
    for word, val in tf_doc.items():
        tf_idf[word] = val * idfs[word]
    return tf_idf

tf_documents = []
for i, doc in enumerate(processed_docs):
    word_counts = Counter(doc)
    tf_doc = compute_tf(word_counts, doc)
    tf_documents.append(tf_doc)
    print(f"\nTerm Frequencies (TF) for Document {i + 1}:")
    for word, tf_value in tf_doc.items():
        print(f"{word}: {tf_value:.4f}")

idf_dict = compute_idf(processed_docs, terms)
print("\nInverse Document Frequencies (IDF):")
for word, idf_value in idf_dict.items():
    print(f"{word}: {idf_value:.4f}")

tf_idf_documents = []
for i, tf_doc in enumerate(tf_documents):
    tf_idf_doc = compute_tf_idf(tf_doc, idf_dict)
    tf_idf_documents.append(tf_idf_doc)
    print(f"\nTF-IDF for Document {i + 1}:")
    for word, tf_idf_value in tf_idf_doc.items():
        print(f"{word}: {tf_idf_value:.4f}")

import pandas as pd
docTermMatrix = pd.DataFrame(tf_idf_documents, columns=terms)
print("\nDocument-Term Matrix (TF-IDF values):")
print(docTermMatrix)
