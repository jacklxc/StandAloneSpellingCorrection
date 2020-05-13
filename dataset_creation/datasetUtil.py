import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from operator import itemgetter, attrgetter
from tqdm import tqdm
from copy import copy
import random
import json

from tokenization import FullTokenizer

def read_berkbeck(path):
    berkbeck = {}
    with open(path,"r") as f:
        for line in f:
            token = line.strip()
            if token[0]=="$":
                correct = token[1:]
                berkbeck[correct] = []
            else:
                berkbeck[correct].append(token)
    return berkbeck

def readOneBillion(path):
    one_billion_files = glob(path+"/*")
    one_billion_sentence_tokens = []
    for filename in sorted(one_billion_files):
        print(filename)
        with open(filename,"r") as f:
            for line in f:
                one_billion_sentence_tokens.append(line.strip().lower())
    return one_billion_sentence_tokens

def adaptERNIEtokenization(all_sentences):
    tokenizer = FullTokenizer(vocab_file="vocab.txt",do_lower_case=True)
    ernie_tokens = [tokenizer.tokenize(sentence) for sentence in tqdm(all_sentences)]
    print("Parsed to ERNIE tokens!")
    all_cleaned_tokens = []
    for line in tqdm(ernie_tokens):
        cleaned_tokens = []
        for i, token in enumerate(line):
            if token[:2] == "##":
                cleaned_tokens[-1] += token[2:]
            else:
                cleaned_tokens.append(token)
        all_cleaned_tokens.append(cleaned_tokens)
    return all_cleaned_tokens

def readYonatan(path):
    natural_replacement = {}
    with open(path,"r") as f:
        for line in f:
            tokens = line.strip().split()
            natural_replacement[tokens[0]] = tokens[1:]
    return natural_replacement

def mergeWordReplacement(dict1, dict2):
    mergedDict = {}
    keys = list(set(dict1.keys()).union(set(dict2.keys())))
    for key in keys:
        mergedDict[key] = list(set(dict1.get(key,[])).union(set(dict2.get(key,[]))))
    return mergedDict

def Swap(word):
    if len(word) > 3:
        i, j = np.random.choice(len(word) - 2, 2, replace=False) + 1
        word = list(word)
        word[i], word[j] = word[j], word[i]
        word = "".join(word)
    return word

def Mid(word):
    if len(word) > 3:
        middle = list(word[1:-1])
        random.shuffle(middle)
        tokens = [word[0]] + middle + [word[-1]]
        word = "".join(tokens)
    return word

def Rand(word):
    word = list(word)
    random.shuffle(word)
    return "".join(word)

def Key(word, key_replacement):
    word = list(word)
    char_index = random.choice(range(len(word)))
    char = word[char_index]
    replace_char = random.choice(key_replacement.get(char.lower(),char))
    word[char_index] = replace_char.upper() if char.isupper() else replace_char
    return "".join(word)

def RandGen(word):
    alphabet = "qwertyuiopasdfghjklzxcvbnm"
    vocab = list(alphabet + alphabet.upper())
    return "".join([random.choice(vocab) for _ in range(len(word))])

def sampleReplacement(replacement, p_train):
    """
    Samples some replacement rules as the known rules.
    """
    train = {}
    for word, candidates in replacement.items():
        for candidate in candidates:
            sample = np.random.choice(2, 1, p=[1-p_train, p_train])
            if sample > 0:
                train_candidates = train.get(word,[])
                train_candidates.append(candidate)
                train[word] = train_candidates
    return train

def messupWord(word, key_replacement):
    funcs = [Swap, Mid, Rand, Key, RandGen]
    func = random.choice(funcs)
    if func == Key:
        return func(word, key_replacement)
    else:
        return func(word)

def charReplacement(sentence_tokens, key_replacement_path = "en.key.txt", word_vocab=None, oov_token="<oov>", sigma=0.2):
    key_replacement = readYonatan(key_replacement_path)
    # Compute Alpha
    N_sentence = len(sentence_tokens)
    normal_numbers = np.random.normal(0,sigma,N_sentence)
    alpha = np.absolute(normal_numbers)
    alpha[alpha > 1] = 1.0

    replaced_one_billion_sentence_tokens = []
    gold_labels = []

    if word_vocab is not None:
        print("Allowing OOV!")
        word_vocab = set(word_vocab)

    for tokens, alp in tqdm(zip(sentence_tokens, alpha)):
        sentence_len = len(tokens)
        m = np.max([np.floor(alp * sentence_len),1]).astype(int)
        replaced_sentence = copy(tokens)
        available_indices = []
        oov_indices = []
        gold_label = {}
        for i, token in enumerate(tokens):
            if word_vocab is not None and token not in word_vocab:
                oov_indices.append(i)
            else:
                available_indices.append(i)
        np.random.shuffle(available_indices)

        if len(available_indices) > 0 or len(oov_indices) > 0:
            for i in available_indices[:m]:
                gold_token = replaced_sentence[i]
                gold_label[i] = gold_token
                replaced_sentence[i] = messupWord(gold_token, key_replacement)
            for i in oov_indices:
                gold_label[i] = oov_token
            gold_labels.append(gold_label)
            replaced_one_billion_sentence_tokens.append(replaced_sentence) # No need to keep sentences with no replacement.
    return replaced_one_billion_sentence_tokens, gold_labels

def wordReplacement(word_replace_vocab, sentence_tokens, word_vocab=None, oov_token="<oov>", sigma=0.2):
    # Compute Alpha
    N_sentence = len(sentence_tokens)
    normal_numbers = np.random.normal(0,sigma,N_sentence)
    alpha = np.absolute(normal_numbers)
    alpha[alpha > 1] = 1.0

    replaced_one_billion_sentence_tokens = []
    gold_labels = []

    if word_vocab is not None:
        print("Allowing OOV!")
        word_vocab = set(word_vocab)

    for tokens, alp in tqdm(zip(sentence_tokens, alpha)):
        sentence_len = len(tokens)
        m = np.max([np.floor(alp * sentence_len),1]).astype(int)
        replaced_sentence = copy(tokens)
        available_indices = []
        oov_indices = []
        gold_label = {}
        for i, token in enumerate(tokens):
            if token in word_replace_vocab:
                available_indices.append(i)
            elif word_vocab is not None and token not in word_vocab:
                # Tricky logic: 
                # if include OOV, for training: should pass word_vocab so that all OOV are tagged with <oov>
                #   for evaluation: should not pass word_vocab so that even OOV can have misspelling, but the model simply cannot deal with it.
                # if not include OOV, sentence_tokens shoud have been filtered already, so no need to pass in the word_vocab.
                oov_indices.append(i) 
        np.random.shuffle(available_indices)

        if len(available_indices) > 0 or len(oov_indices) > 0:
            for i in available_indices[:m]:
                gold_token = replaced_sentence[i]
                gold_label[i] = gold_token
                replaced_sentence[i] = random.choice(word_replace_vocab[gold_token])
            for i in oov_indices:
                gold_label[i] = oov_token
            gold_labels.append(gold_label)
            replaced_one_billion_sentence_tokens.append(replaced_sentence) # No need to keep sentences with no replacement.
    return replaced_one_billion_sentence_tokens, gold_labels

def sortByCount(array, sort=True):
    """
    Given a list, count the number of each element and sort them according to the most frequent to the least frequent.
    Returns a list of tuples, (item, count).
    """
    array_dict = {}
    for item in list(set(array)):
        array_dict[item] = 0
    for item in array:
        array_dict[item] += 1
    sorted_list = [(item, count) for item, count in array_dict.items()]
    if sort:
        sorted_list = sorted(sorted_list, key=itemgetter(1), reverse=True)
    return sorted_list

def mergeCount(count_list):
    array_dict = {}
    for counts in count_list:
        for word, count in counts:
            prev_count = array_dict.get(word,0)
            array_dict[word] = prev_count + count
    sorted_list = [(item, count) for item, count in array_dict.items()]
    sorted_list = sorted(sorted_list, key=itemgetter(1), reverse=True)
    return sorted_list

def countCorpus(one_billion_sentence_tokens):
    all_counts = []
    for sentence in tqdm(one_billion_sentence_tokens):
        all_counts.append(sortByCount(sentence,sort=False))
    mergedCounts = mergeCount(all_counts)
    return mergedCounts

def countChar(wordCounts):
    charCounts = {}
    for word, count in wordCounts:
        chars = list(word)
        for char in chars:
            charCounts[char] = charCounts.get(char,0) + count
    sorted_list = [(item, count) for item, count in charCounts.items()]
    sorted_list = sorted(sorted_list, key=itemgetter(1), reverse=True)
    return sorted_list

def charVocab(wordCounts):
    all_char = ""
    for word, count in wordCounts:
        all_char+=word
    return list(set(all_char))

def excludeSentenceWithUnknownWords(wordVocab, one_billion_sentence_tokens):
    filtered_one_billion_sentence_tokens = []
    OOV_sentence_tokens = []
    wordVocab = set(wordVocab)
    for sentence in one_billion_sentence_tokens:
        valid = True
        for token in sentence:
            if token not in wordVocab:
                valid = False
                break
        if valid:
            filtered_one_billion_sentence_tokens.append(sentence)
        else:
            OOV_sentence_tokens.append(sentence)
    return filtered_one_billion_sentence_tokens, OOV_sentence_tokens

def sentence2examples(replaced_tokens, labels, correct_labels, unk = "<unk>"):
    """
    Convert one sentence with multiple replacements to multiple examples with corresponding replacements.
    Returns:
    - texts: [(unk_replaced_tokens, correct_token, positional_index), (...), ...]
    - individual_labels: list of labels
    """
    texts = []
    individual_labels = []
    tokens_unk = copy(replaced_tokens)
    for index, label in labels.items():
        tokens_unk[index] = unk
    labels.update(correct_labels)
    for index, label in labels.items():
        texts.append((tokens_unk,replaced_tokens[index], index))
        individual_labels.append(label)
    return texts, individual_labels

def all_sentence2examples(all_replaced_tokens, all_labels, all_correct_labels=None, unk = "<unk>"):
    all_texts = []
    individual_labels = []
    for i, (sentence, labels) in enumerate(zip(all_replaced_tokens, all_labels)):
        if all_correct_labels is not None:
            correct_labels = all_correct_labels[i]
        else:
            correct_labels = set([])
        texts, individual_label = sentence2examples(sentence, labels, correct_labels, unk = unk)
        all_texts.extend(texts)
        individual_labels.extend(individual_label)
    return all_texts, individual_labels

def saveDataset(path, texts, labels):
    with open(path, "w") as f:
        for (sentence, mistake, index), label in tqdm(zip(texts, labels)):
            f.write(" ".join(sentence)+"\t"+mistake+"\t"+str(index)+"\t"+label+"\n")
    print("Done!")

def saveNoisyInput(path, sentence_tokens):
    with open(path, "w") as f:
        for tokens in sentence_tokens:
            f.write(" ".join(tokens)+"\n")

def saveNoisyInputLabel(path, gold_labels):
    with open(path, "w") as f:
        for label in gold_labels:
            json.dump(label,f)
            f.write("\n")

def loadNoisyInputLabel(path):
    loaded_labels = []
    with open(path,"r") as f:
        for line in f:
            loaded_label = json.loads(line)
            loaded_labels.append({int(index): label for index, label in loaded_label.items()})
    return loaded_labels

def readVocab(char_path, word_path):
    char_vocab = {}
    word_vocab = {}
    with open(char_path,"r") as f:
        count = 0
        for line in f:
            count += 1
            char = line[0]
            char_vocab[char] = count
    with open(word_path,"r") as f:
        count = 0
        for line in f:
            count += 1
            word = line.strip()
            word_vocab[word] = count
    return char_vocab, word_vocab

def mixNormalWords(sentence_tokens, original_labels, sigma = 0.2):
    # Compute Alpha
    correct_labels = []
    N_sentence = len(sentence_tokens)
    normal_numbers = np.random.normal(0,sigma,N_sentence)
    alpha = np.absolute(normal_numbers)
    alpha[alpha > 1] = 1.0
    
    for j, (tokens, labels, alp) in tqdm(enumerate(zip(sentence_tokens, original_labels, alpha))):
        correct_label = {}
        sentence_len = len(tokens)
        m = np.max([np.floor(alp * sentence_len),1]).astype(int)
        available_indices = []
        gold_label = {}
        for i, token in enumerate(tokens[:-1]):
            if i not in labels.keys():
                available_indices.append(i)
        np.random.shuffle(available_indices)
        if len(available_indices) > 0:
            for i in available_indices[:m]:
                gold_token = tokens[i]
                correct_label[i] = gold_token
        correct_labels.append(correct_label)
    return correct_labels

def filterOutlierSentence(one_billion_sentence_tokens, max_sentence_len=200, max_word_len=20):
    filtered_one_billion_sentence_tokens = []
    for tokens in one_billion_sentence_tokens:
        if len(tokens) <= max_sentence_len:
            token_lens = [len(token) for token in tokens]
            if max(token_lens) <= max_word_len:
                filtered_one_billion_sentence_tokens.append(tokens)
    return filtered_one_billion_sentence_tokens