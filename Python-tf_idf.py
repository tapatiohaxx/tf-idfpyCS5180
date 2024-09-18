#-------------------------------------------------------------------------
# AUTHOR: Matthew Plascencia
#BroncoID: 012600809
# FILENAME: Python-tf-idf.py
# SPECIFICATION: A python program to do the work for problem 7:It foes the stopword removal (removes pronouns/conjugations/etc)
# and then does stemming. After that it indexes the words and does TF, IDF, and TF-IDF. It prints them in easy to read tables and matrices.
# FOR: CS 5180- Assignment #1
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

import csv
import math

documents = []
#Step 0: read in everything from the CSV file

with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append(row[0])

#Step 1: conduct stopword removal and stemming
stopWords = {"i", "and", "she", "her", "they", "their", "the"}
steeming = {
    "loves": "love",
    "dogs": "dog",
    "cats": "cat"
}
def preprocess(document):
    words = document.lower().split()  # Split words in document
    processed_words = []
    for word in words:
        if word not in stopWords:  # Remove stopwords
            # Apply stemming if the word is in the stemming dictionary
            stemmed_word = steeming.get(word, word)
            processed_words.append(stemmed_word)
    return processed_words
processed_docs = [preprocess(doc) for doc in documents]

terms = list(set([term for doc in processed_docs for term in doc]))

#Step 2: Calculate the TF and IDF. Show all work.
def compute_tf(doc):
    tf_dict = {}
    total_terms = len(doc)
    for word in doc:
        tf_dict[word] = tf_dict.get(word, 0) + 1

    print(f"\nCalculating TF for document: {doc}")
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_terms
        print(f"TF({word}) = {tf_dict[word]} (word count / total terms = {tf_dict[word]} / {total_terms})")
    return tf_dict

def compute_idf(doc_list, vocab):
    N = len(doc_list)
    idf_dict = dict.fromkeys(vocab, 0)  # Initialize IDF dictionary for all words in vocab

    print("\nCalculating IDF for each word:")
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] += 1

    for word, count in idf_dict.items():
        idf_value = math.log10(N / float(count))
        idf_dict[word] = idf_value
        print(f"IDF({word}) = log10(N / df) = log10({N} / {count}) = {idf_value}")

    return idf_dict
#Step 3: Calculate TF-IDF
def compute_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = {}

    print("\nCalculating TF-IDF for document:")
    for word, tf_value in tf_dict.items():
        tf_idf_value = tf_value * idf_dict.get(word, 0)
        tf_idf_dict[word] = tf_idf_value
        print(f"TF-IDF({word}) = TF({word}) * IDF({word}) = {tf_value} * {idf_dict.get(word, 0)} = {tf_idf_value}")

    return tf_idf_dict
#Step 4: Make the TF-IDF matrix.
docTermMatrix = []
tf_documents = [compute_tf(doc) for doc in processed_docs]
idf_dict = compute_idf(processed_docs, terms)
tf_idf_documents = [compute_tf_idf(tf_doc, idf_dict) for tf_doc in tf_documents]

print("\nDocument-Term Matrix (TF-IDF values):")

header = ["Document"] + terms
print(f"{'Document':<10}", end="")
for term in terms:
    print(f"{term:<10}", end="")
print()

for i, tf_idf_doc in enumerate(tf_idf_documents, 1):
    print(f"{'Doc ' + str(i):<10}", end="")
    for term in terms:
        # Get TF-IDF value or 0 if term is not present in the document
        tf_idf_value = tf_idf_doc.get(term, 0)
        print(f"{tf_idf_value:<10.4f}", end="")
    print()  # New line for next document
