import numpy as np
import re
import collections

feature_label = 0
lang_id = 0
feature_dict = {}
language_id_dict = {}
word_counts = {}

def read_data(filename='Corpus.txt', return_unlabeled=False):
    output_array = []
    regex = r'(.*)([^-]\w{2}|\w{2}-\w{2})'
    
    with open(filename, 'r', encoding="UTF-16-LE") as f:
        for line in f:    
            output_line = ''

            match = re.search(regex, line)
            sentence = "".join(match.group(1))
            label = match.group(2).strip()

            sentence_features = {}
            output_line += get_language_id(label)
            
            sentence_features = char_feature(sentence, sentence_features)
            sentence_features = word_count(sentence, sentence_features)
            sentence_features = sentence_length(sentence, sentence_features)
            sentence_features = bigrams(sentence, sentence_features)

            output_line += get_line_from_features(sentence_features)
            output_array.append(output_line)
    
    return output_array


def get_line_from_features(features):
    ordered_dict = collections.OrderedDict(sorted(features.items()))

    line = ''
    for k, v in ordered_dict.items():
        line += ' ' + str(k) + ':' + str(v)

    return line

def get_language_id(label):
    global lang_id
    if label in language_id_dict:
        return str(language_id_dict.get(label))
    else:
        lang_id += 1
        language_id_dict[label] = lang_id
        return str(lang_id)

def char_feature(sentence, sentence_features):
    global feature_label
    for c in sentence:
        if c in feature_dict:
            label = feature_dict.get(c)
            if c not in sentence_features:
                sentence_features[label] = 1
        else:
            feature_label += 1
            feature_dict[c] = feature_label
            sentence_features[feature_label] = 1 
    return sentence_features

def word_count(sentence, sentence_features):
    global feature_label
    words = sentence.split()
    if len(words) in feature_dict:
        label =  feature_dict.get(len(words))
        sentence_features[label] = 1
    else:
        feature_label += 1
        feature_dict[len(words)] = feature_label
        sentence_features[feature_label] = 1
    return sentence_features

def sentence_length(sentence, sentence_features):
    global feature_label
    s = "".join(sentence.split())
    if len(s) in feature_dict:
        label = feature_dict.get(len(s))
        sentence_features[label] = 1
    else:
        feature_label += 1
        feature_dict[len(s)] = feature_label
        sentence_features[feature_label] = 1

    return sentence_features

def bigrams(sentence, sentence_features):
    global feature_label
    s = "".join(sentence.split())
    bigram_list = [s[i:i+2] for i in range(len(s)-1)]
    for bi in bigram_list:
        if bi in feature_dict:
            label = feature_dict.get(bi)
            if bi not in sentence_features:
                sentence_features[label] = 1
        else:
            feature_label += 1
            feature_dict[bi] = feature_label
            sentence_features[feature_label] = 1
    return sentence_features

def write_array(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            line += '\n'
            f.write(line)

data = read_data()

np.random.shuffle(data)
train_data = data[:int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]

write_array(train_data, "train.txt")
write_array(test_data, "test.txt")

