#!/usr/bin/env python
# coding: utf-8

from datasetUtil import *

prefix = "data/"
input_path = prefix+"train_mix.txt"
label_path = prefix+"train_mix_label.jsonl"
output_path = prefix+"train_mix_ernie.tsv"

separator = " "

with open(input_path, "r") as f:
    with open(label_path, "r") as lf:
        with open(output_path,"w") as wf:
            wf.write("text_a\tlabel\n")
            for line, label_line in zip(f, lf):
                tokens = line.strip().split()
                label = eval(label_line.strip())
                label = {int(k):v for k, v in label.items()}
                
                correct_tokens = ["B-"+token for token in tokens]
                for index, correct in label.items():
                    correct_tokens[index] = "B-"+correct
                wf.write(separator.join(tokens)+"\t"+separator.join(correct_tokens)+"\n")

char_path = prefix+"charVocab.txt"
word_path = prefix+"wordVocab.txt"
charList, wordList = readVocab(char_path, word_path)

mapping = {}
for word, index in wordList.items():
    mapping["B-"+word] = index * 2 - 1
    mapping["I-"+word] = index * 2
mapping["O"] = index * 2 + 1

print("Full label map length:",len(mapping))

with open(prefix+"full_label_map.json","w") as f:
    json.dump(mapping,f)

