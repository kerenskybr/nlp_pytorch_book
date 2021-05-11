# Tokenizing text
import spacy
nlp = spacy.load('en')
text = "Mary, don’t slap the green witch"
print([str(token) for token in nlp(text.lower())])

print("*"*30)

from nltk.tokenize import TweetTokenizer
tweet=u"Snow White and the Seven Degrees"
#MakeAMovieCold@midnight:­)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))

# Feature enginnering
# Unigrams, Bigrams, Trigrams, ..., N-grams
def n_grams(text, n):
    '''
    takes tokens or text, returns a list of n­grams
    '''
    return [text[i:i+n] for i in range (len(text) - n + 1)]

cleaned = ['mary', ',', "n't", 'slap', 'green', 'witch', '.']
print(n_grams(cleaned, 3))

# Lemmas and Stems
import spacy
nlp = spacy.load('en')
doc = nlp(u"he was running late")
for token in doc:
    print('{} ­­> {}'.format(token, token.lemma_))

# Categorizing Words: POS Tagging
nlp = spacy.load('en')
doc = nlp(u"Mary slapped the green witch.")
for token in doc:
    print('{} ­ {}'.format(token, token.pos_))

# Noun Phrase (NP) chunking
nlp = spacy.load('en')
doc = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print ('{} ­ {}'.format(chunk, chunk.label_))