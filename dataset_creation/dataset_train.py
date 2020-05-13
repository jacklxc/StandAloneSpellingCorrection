from datasetUtil import *
import json
import pickle

prefix = "data/"
one_billion_path = "1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
one_billion_sentence_tokens = readOneBillion(one_billion_path)
print(len(one_billion_sentence_tokens))
#mergedCounts = countCorpus(one_billion_sentence_tokens)

with open("mergedCounts_train.pkl", "rb") as f:
    mergedCounts = pickle.load(f)

frequentWordCount = mergedCounts[:50000]
wordList = [word for word, count in frequentWordCount]
charList = charVocab(frequentWordCount)
#with open(prefix+"wordVocab.txt","w") as f:
#    for word in wordList:
#        f.write(word+"\n")
#    f.write("<unk>"+"\n")
#    f.write("<oov>"+"\n")
#with open(prefix+"charVocab.txt","w") as f:
#    for char in charList:
#        f.write(char+"\n")

oov_one_billion_sentence_tokens = filterOutlierSentence(one_billion_sentence_tokens)
print("Filtered too long sentences.",len(oov_one_billion_sentence_tokens))

N_test = 10000
test_sentences = oov_one_billion_sentence_tokens[:N_test]
dev_sentences = oov_one_billion_sentence_tokens[N_test:2*N_test]
train_sentences = oov_one_billion_sentence_tokens[2*N_test:]

filtered_train, _ = excludeSentenceWithUnknownWords(wordList, train_sentences)
filtered_dev, _ = excludeSentenceWithUnknownWords(wordList, dev_sentences)
filtered_test, _ = excludeSentenceWithUnknownWords(wordList, test_sentences)

#print("Train", len(train_sentences), len(filtered_train))
#print("Dev", len(dev_sentences), len(filtered_dev))
#print("Test", len(test_sentences), len(filtered_test))

#with open(prefix+"train1B.txt","w") as f:
#    for sentence in train_sentences:
#        f.write(" ".join(sentence)+"\n")
#with open(prefix+"dev1B.txt","w") as f:
#    for sentence in dev_sentences:
#        f.write(" ".join(sentence)+"\n")
#with open(prefix+"test1B.txt","w") as f:
#    for sentence in test_sentences:
#        f.write(" ".join(sentence)+"\n")

berkbeck = read_berkbeck("missp.dat.txt")
natural_replacement = readYonatan("en.natural.txt")
mergedWordReplacement = mergeWordReplacement(berkbeck, natural_replacement)

with open("train_replacement.json") as f:
    trainWordReplacement = json.load(f)

#replaced_one_billion_sentence_tokens, gold_labels = wordReplacement(trainWordReplacement, filtered_train, word_vocab = None, sigma=0.2)
#saveNoisyInput(prefix+"train.txt", replaced_one_billion_sentence_tokens)
#saveNoisyInputLabel(prefix+"train_label.jsonl", gold_labels)

oov_replaced_one_billion_sentence_tokens, oov_gold_labels = wordReplacement(trainWordReplacement, train_sentences, word_vocab = wordList, sigma=0.1)
saveNoisyInput(prefix+"oov_train.txt", oov_replaced_one_billion_sentence_tokens)
saveNoisyInputLabel(prefix+"oov_train_label.jsonl", oov_gold_labels)

#replaced_one_billion_sentence_tokens, gold_labels = wordReplacement(trainWordReplacement, filtered_dev, word_vocab = None, sigma=0.2)
#saveNoisyInput(prefix+"dev.txt", replaced_one_billion_sentence_tokens)
#saveNoisyInputLabel(prefix+"dev_label.jsonl", gold_labels)

oov_replaced_one_billion_sentence_tokens, oov_gold_labels = wordReplacement(trainWordReplacement, dev_sentences, word_vocab = wordList, sigma=0.1)
saveNoisyInput(prefix+"oov_dev.txt", oov_replaced_one_billion_sentence_tokens)
saveNoisyInputLabel(prefix+"oov_dev_label.jsonl", oov_gold_labels)

#replaced_one_billion_sentence_tokens, gold_labels = wordReplacement(mergedWordReplacement, filtered_test, word_vocab = None, sigma=0.2)
#saveNoisyInput(prefix+"test.txt", replaced_one_billion_sentence_tokens)
#saveNoisyInputLabel(prefix+"test_label.jsonl", gold_labels)

oov_replaced_one_billion_sentence_tokens, oov_gold_labels = wordReplacement(mergedWordReplacement, test_sentences, word_vocab = wordList, sigma=0.1)
saveNoisyInput(prefix+"oov_test.txt", oov_replaced_one_billion_sentence_tokens)
saveNoisyInputLabel(prefix+"oov_test_label.jsonl", oov_gold_labels)