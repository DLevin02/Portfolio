## Written by: Drew Levin



import string
import re
from collections import Counter
from itertools import product
from itertools import permutations
import random
from numpy import cumsum
import numpy as np

# adjust on your own
P_my = 0.87
P_fake = 0.13
num_charactors = 1000

with open('script.txt', encoding = 'utf-8') as f:
    data = f.read()

def process_text(data):
    data = data.lower()
    data = re.sub(r'[^a-z ]+', '', data)
    data = ' '.join(data.split())
    return data
data = process_text(data)

allchar = ' ' + string.ascii_lowercase
unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch]/len(data),4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

# to distinguish between fake_unigram_prob below
my_unigram_prob = unigram_prob

def ngram(n):
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(allchar, repeat=n)],0)
    # update counts
    d.update(Counter(data[x:x+n] for x in range(len(data)-1)))
    return d

bigram = ngram(2)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}
bigram_prob_L = {
    c: (bigram[c] + 1) / (unigram[c[0]] + 27) for c in bigram}

trigram= ngram(3)
trigram_prob_L = {c: (trigram[c] + 1) / (bigram[c[:2]] + 27) for c in trigram}

# based on https://python-course.eu/numerical-programming/weighted-probabilities.php
def weighted_choice(collection, weights):
    weights = np.array(weights)
    weights_sum = weights.sum()
    weights = weights.cumsum()/weights_sum
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return collection[i]

def gen_bi(c):
    w = [bigram_prob[c+i] for i in allchar]
    return weighted_choice(allchar, weights = w)[0]

def gen_tri(ab):
    w = [trigram_prob_L[ab+i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]

def gen_sen(c, num):
    # generate the second char
    res = c + gen_bi(c)
    for i in range(num-2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


sentences = []
for char in allchar:
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

## fake script
with open('script1.txt', encoding = 'utf-8') as f:
    data = f.read()

data = process_text(data)

unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch]/len(data),4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

fake_unigram_prob = unigram_prob

count = 0
for char in allchar:
    count += 1
    x = (P_fake*fake_unigram_prob[char]/(P_fake*fake_unigram_prob[char] + P_my*my_unigram_prob[char]))
    print(round(x, 4))


for sentence in sentences:
    my = 0
    fake = 0
    for char in sentence:
        my += np.log10(my_unigram_prob[char])
        fake += np.log10(fake_unigram_prob[char])
    if my > fake:
        print('0')
    else:
        print('1')