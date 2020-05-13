from datasetUtil import *
import json
import pickle

prefix = "data/"
one_billion_path = "1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
one_billion_sentence_tokens = readOneBillion(one_billion_path)
print(len(one_billion_sentence_tokens))

char_path = prefix + "charVocab.txt"
word_path = prefix + "wordVocab.txt"
charList, wordList = readVocab(char_path, word_path)

oov_one_billion_sentence_tokens = filterOutlierSentence(one_billion_sentence_tokens)
print("Filtered too long sentences.",len(oov_one_billion_sentence_tokens))

N_test = 10000
test_sentences = oov_one_billion_sentence_tokens[:N_test]
dev_sentences = oov_one_billion_sentence_tokens[N_test:2*N_test]
train_sentences = oov_one_billion_sentence_tokens[2*N_test:]

filtered_train, _ = excludeSentenceWithUnknownWords(wordList, train_sentences)
filtered_dev, _ = excludeSentenceWithUnknownWords(wordList, dev_sentences)
filtered_test, _ = excludeSentenceWithUnknownWords(wordList, test_sentences)

print("Train", len(train_sentences), len(filtered_train))
print("Dev", len(dev_sentences), len(filtered_dev))
print("Test", len(test_sentences), len(filtered_test))

with open(prefix + "train1B_excludeOOV.txt","w") as f:
    for sentence in filtered_train:
        f.write(" ".join(sentence)+"\n")
with open(prefix + "dev1B_excludeOOV.txt","w") as f:
    for sentence in filtered_dev:
        f.write(" ".join(sentence)+"\n")
with open(prefix + "test1B_excludeOOV.txt","w") as f:
    for sentence in filtered_test:
        f.write(" ".join(sentence)+"\n")

replaced_one_billion_sentence_tokens, gold_labels = charReplacement(filtered_train, word_vocab = None, sigma=0.2)
saveNoisyInput(prefix + "char_train.txt", replaced_one_billion_sentence_tokens)
saveNoisyInputLabel(prefix + "char_train_label.jsonl", gold_labels)

oov_replaced_one_billion_sentence_tokens, oov_gold_labels = charReplacement(train_sentences, word_vocab = wordList, sigma=0.1)
saveNoisyInput(prefix + "oov_char_train.txt", oov_replaced_one_billion_sentence_tokens)
saveNoisyInputLabel(prefix + "oov_char_train_label.jsonl", oov_gold_labels)