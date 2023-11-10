# %%
import os
import collections

import torchtext

# %%
os.makedirs('./data', exist_ok=True)
# dataディレクトリにAG_NEWSデータセットをダウンロード
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

# %%
# tokenizer: 文章を単語に分割する
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# %%
first_sentence = train_dataset[0][1]

f_tokens = tokenizer(first_sentence)
# 29 ['wall', 'st', '.', 'bears', 'claw', 'back', 'into', 'the', 'black', '(', 'reuters', ')', 'reuters', '-', 'short-sellers', ',', 'wall', 'street', "'", 's', 'dwindling\\band', 'of', 'ultra-cynics', ',', 'are', 'seeing', 'green', 'again', '.']

# %%
# データセット内の各単語の出現回数をカウント
counter = collections.Counter()
for _, line in train_dataset:
    counter.update(tokenizer(line))

# %%
# vocab: データセットに出現する単語を単語IDとして持つ。
vocab = torchtext.vocab.vocab(counter, min_freq=1)
vocab_size = len(vocab)

# %%
word_lookup = [list((vocab[w], w)) for w in f_tokens]
# [[0, 'wall'], [1, 'st'], [2, '.'], [3, 'bears'], [4, 'claw'], [5, 'back'], [6, 'into'], [7, 'the'], [8, 'black'], [9, '('], [10, 'reuters'], [11, ')'], [10, 'reuters'], [12, '-'], [13, 'short-sellers'], [14, ','], [0, 'wall'], [15, 'street'], [16, "'"], [17, 's'], [18, 'dwindling\\band'], [19, 'of'], [20, 'ultra-cynics'], [14, ','], [21, 'are'], [22, 'seeing'], [23, 'green'], [24, 'again'], [2, '.']]

# %%
def encode(x):
    return [vocab.get_stoi()[s] for s in tokenizer(x)]

# 文章中の単語を単語IDに変換
vec = encode(first_sentence)
# wall -> 0
# st -> 1
# . -> 2
# bears -> 3
# claw -> 4
# back -> 5

# %%
def decode(x):
    return [vocab.get_itos()[i] for i in x]

# 単語IDを単語に変換
decoded_vec = decode(vec)
# 0 -> wall
# 1 -> st
# 2 -> .
# 3 -> bears
# 4 -> claw
# 5 -> back

# %%
from torchtext.data.utils import ngrams_iterator

# 'hot dog'という単語列の意味と'hot'と'dog'の単語の意味は意味が全く異なる。
# そこで、bivocab(N-grameでn=2)が全ての単語のペアを格納する

bi_counter = collections.Counter()
for label, line in train_dataset:
    bi_counter.update(ngrams_iterator(tokenizer(line), ngrams=2))
bi_vocab = torchtext.vocab.vocab(bi_counter, min_freq=2)

# %%
# 単語IDに変換
def encode(x):
    bi_vocabs = []
    for s in tokenizer(x):
        if counter[s] == 1:
            continue
        bi_vocabs.append(bi_vocab.get_stoi()[s])
    return bi_vocabs


