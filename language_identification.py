import re
import argparse
import numpy as np
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename', required=True, help='Path to corpus')
ap.add_argument('-s', '--seed', type=int, default=53, help='Seed for random number generation')
ap.add_argument('-o', '--output', help='Path to output statistics. If given, statistics are written to a file')
args = vars(ap.parse_args())
np.random.seed(args['seed'])

UNKNOWN_LETTER = "<UNK>"


def read_data(filename='devel.txt', return_unlabeled=False):
    train_data = []
    train_labels = []

    unlabeled_data = []

    with open(filename, 'r', encoding="UTF-16-LE") as f:
        train_count = 0
        unlabeled_count = 0
        regex = r'(.*)([^-]\w{2}|\w{2}-\w{2})'
        for line in f:
            match = re.search(regex, line)
            sentence = "".join(match.group(1).split())
            label = match.group(2).strip()

            train_count += 1
            train_data.append(sentence)
            train_labels.append(label)

        print(f"Total instances: {train_count}")

    if return_unlabeled:
        return train_data, train_labels, unlabeled_data
    else:
        return train_data, train_labels


def split_train_test(data, labels, test_split=0.1):
    count = len(data)
    data = np.array(data)
    labels = np.array(labels)
    shuffled_indices = np.random.permutation(np.arange(count))
    train_indices = shuffled_indices[:int(count * (1 - test_split))]
    test_indices = shuffled_indices[int(count * (1 - test_split)):]
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    return train_data, test_data, train_labels, test_labels


def train_bayes(train_data, train_labels):
    language_statistics = {}
    all_letters = []
    for i in range(len(train_data)):
        sentence = train_data[i]
        label = train_labels[i]
        letters = list(sentence)
        letters = np.array(letters)
        letters, counts = np.unique(letters, return_counts=True)
        all_letters.extend(letters)
        language_statistics[label] = language_statistics.get(label, {})
        for l, c in zip(letters, counts):
            language_statistics[label][l] = language_statistics[label].get(l, 0) + c

    # Statistics with Laplace Smoothing
    all_letters = np.array(all_letters)
    all_letters, letter_counts = np.unique(all_letters, return_counts=True)
    for k, v in language_statistics.items():
        total_letter_count = np.sum(list(language_statistics[k].values()))
        for l, c in language_statistics[k].items():
            language_statistics[k][l] = (language_statistics[k][l] + 1) / (total_letter_count + len(all_letters))
        language_statistics[k][UNKNOWN_LETTER] = 1 / (total_letter_count + len(all_letters))
    all_letters = dict(zip(all_letters, letter_counts))
    return language_statistics, all_letters


def calculate_statistics(languages, y_predicted, y_expected):
    statistics = {}
    y_predicted = np.array(y_predicted)
    y_expected = np.array(y_expected)
    for language in languages:
        statistics[language] = {}
        true_positive = np.sum(y_predicted[y_expected == language] == language)
        statistics[language]["TP"] = true_positive
        true_negative = np.sum(y_predicted[y_expected != language] != language)
        statistics[language]["TN"] = true_negative
        false_positive = np.sum(y_predicted[y_expected != language] == language)
        statistics[language]["FP"] = false_positive
        false_negative = np.sum(y_predicted[y_expected == language] != language)
        statistics[language]["FN"] = false_negative
        precision = true_positive / (true_positive + true_negative)
        recall = true_positive / (true_positive + false_negative)
        f_score = (2 * precision * recall) / (precision + recall)
        statistics[language]["Precision"] = precision
        statistics[language]["Recall"] = recall
        statistics[language]["F-measure"] = f_score

    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0
    for language in languages:
        sum_tp += statistics[language]["TP"]
        sum_tn += statistics[language]["TN"]
        sum_fp += statistics[language]["FP"]
        sum_fn += statistics[language]["FN"]

    statistics["overall"] = {}
    statistics["overall"]["TP"] = sum_tp
    statistics["overall"]["TN"] = sum_tn
    statistics["overall"]["FP"] = sum_fp
    statistics["overall"]["FN"] = sum_fn
    statistics["overall"]["Precision"] = sum_tp / (sum_tp + sum_tn)
    statistics["overall"]["Recall"] = sum_tp / (sum_tp + sum_fn)
    statistics["overall"]["F-measure"] = (2 * statistics["overall"]["Precision"] * statistics["overall"]["Recall"]) / (
            statistics["overall"]["Precision"] + statistics["overall"]["Recall"])

    return statistics


def print_statictics(statistics, filename):
    with open(filename, "w") as f:
        for language, statistic in statistics.items():
            f.write(f"{language}:\n")
            for s, v in statistics[language].items():
                f.write(f"\t{s}: {v}\n")


data, labels = read_data(filename=args['filename'])
train_data, test_data, train_labels, test_labels = split_train_test(data, labels)

language_statistics, all_letters = train_bayes(train_data, train_labels)
languages = list(language_statistics.keys())

predictions = []
for sentence in test_data:
    probabilities = {}
    letters = list(sentence)
    for language in languages:
        probabilities[language] = 1.0
        for letter in letters:
            probabilities[language] *= language_statistics[language].get(letter,
                                                                         language_statistics[language][UNKNOWN_LETTER])

    predicted_class = languages[0]
    max_prob = 0
    for k, v in probabilities.items():
        if v > max_prob:
            max_prob = v
            predicted_class = k
    predictions.append(predicted_class)

true = 0
for a, b in zip(predictions, test_labels):
    if a == b:
        true += 1

statistics = calculate_statistics(languages, predictions, test_labels)

if args["output"]:
    print_statictics(statistics, args["output"])

confusion_matrix = ConfusionMatrix(test_labels, predictions)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot()
plt.show()
