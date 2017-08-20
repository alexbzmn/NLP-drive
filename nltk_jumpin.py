from nltk import word_tokenize, pos_tag, ne_chunk

text = "John works at Intel."  # str

# Morphology Level
tokens = word_tokenize(text)
print tokens  # [str]
# ['John', 'works', 'at', 'Intel', '.']

# Syntax Level
tagged_tokens = pos_tag(tokens)
print tagged_tokens  # [(str, tag)]
# [('John', 'NNP'), ('works', 'VBZ'), ('at', 'IN'), ('Intel', 'NNP'), ('.', '.')]

# Semantics Level
ner_tree = ne_chunk(tagged_tokens)
print ner_tree  # nltk.Tree
# (S (PERSON John/NNP) works/VBZ at/IN (ORGANIZATION Intel/NNP) ./.)


from nltk import bigrams, trigrams

text = "John works at Intel."
tokens = word_tokenize(text)

print list(bigrams(tokens))  # the `bigrams` function returns a generator, wo we must unwind it
# [('John', 'works'), ('works', 'at'), ('at', 'Intel'), ('Intel', '.')]

print list(trigrams(tokens))  # the `trigrams` function returns a generator, wo we must unwind it
# [('John', 'works', 'at'), ('works', 'at', 'Intel'), ('at', 'Intel', '.')]

#########################################################
from nltk import Text
from nltk.corpus import reuters

text = Text(reuters.words())
# Get the collocations that don't contain stop-words
text.collocations()
# United States; New York; per cent; Rhode Island; years ago; Los Angeles; White House; ...
# Get words that appear in similar contexts
text.similar('Monday', 5)
# april march friday february january
# Get common contexts for a list of words
text.common_contexts(['August', 'June'])
# since_a in_because last_when between_and last_that and_at ...
# Get contexts for a word
text.concordance('Monday')

#########################################################

from nltk.corpus import webtext
from nltk import word_tokenize
from nltk import FreqDist

# Build a large text
text = ""
for wt in webtext.fileids()[:100]:
    text += "\n\n" + webtext.raw(wt)

fdist = FreqDist(word_tokenize(text))

# Get the text's vocabulary
print fdist.keys()[:100]  # First 100 words

print fdist['dinosaurs']  # 7

# Get a word's frequency
print fdist.freq('dinosaurs')  # 1.84242041402e-05
# Total number of samples
print fdist.N()  # 379935
# Words that appear exactly once
print fdist.hapaxes()  # [u'sepcially', u'mutinied', u'Nudists', u'Restrained', ... ]
# Most common samples
print fdist.most_common(n=5)  # [(u'.', 16500), (u':', 14327), (u',', 12427), (u'I', 7786), (u'the', 7313)]
# Draw a bar chart with the count of the most common 50 words
import matplotlib.pyplot as plt

x, y = zip(*fdist.most_common(n=50))
plt.bar(range(len(x)), y)
plt.xticks(range(len(x)), x)
plt.show()

#####################################

import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
## Bigrams
finder = BigramCollocationFinder.from_words(nltk.corpus.reuters.words())
# only bigrams that appear 5+ times
finder.apply_freq_filter(5)
# return the 50 bigrams with the highest PMI
print finder.nbest(bigram_measures.pmi, 50)
# among the collocations we can find stuff like: (u'Corpus', u'Christi'), (u'mechanically', u'separated'), (u'Kuala', u'Lumpur'), (u'Mathematical', u'Applications')
## Trigrams
finder = TrigramCollocationFinder.from_words(nltk.corpus.reuters.words())
# only trigrams that appear 5+ times
finder.apply_freq_filter(5)
# return the 50 trigrams with the highest PMI
print finder.nbest(trigram_measures.pmi, 50)
# among the collocations we can find stuff like: (u'GHANA', u'COCOA', u'PURCHASES'), (u'Punta', u'del', u'Este'), (u'Special', u'Drawing', u'Rights')
