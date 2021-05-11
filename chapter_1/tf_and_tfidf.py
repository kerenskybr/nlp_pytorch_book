from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# TF (Term­Frequency) example
corpus = ['Time flies flies like an arrow.',
        'Fruit flies like a banana.']

# Get tokens to display labels
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()

# Encode to display encoded
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
    cbar=False, xticklabels=vocab,
    yticklabels=['Sentence 1', 'Sentence 2'])
plt.show()

# TF-IDF (Term­Frequency­Inverse­Document­Frequency) example
# The IDF representation penalizes common tokens and rewards rare tokens in the vector representation.
# IDF(w) = log * N / nw
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab,
    yticklabels= ['Sentence 1', 'Sentence 2'])
plt.show()
