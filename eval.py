##
##  Class for evaluationg and get statistics of SVM results
##
##
import numpy as np
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

y_predicted = []
y_expected = []


with open('output.txt', 'r') as f:
    for line in f:
        cols = line.split()
        y_predicted.append(cols[0])  

with open('test.txt', 'r') as f:
    for line in f:
        cols = line.split()
        y_expected.append(cols[0])

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
        if (precision+recall) == 0:
            f_score = 'unknown'
        else:
            f_score = (2*precision*recall)/(precision + recall)
        print((precision + recall))
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


lang = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr']

y_p = []
y_e = []
for y in y_expected:
    y_e.append(lang[int(y)-1])

for y in y_predicted:
    y_p.append(lang[int(y)-1])

print(y_e)

stat = calculate_statistics(lang, y_p, y_e)
print_statictics(stat, "stat.txt")



confusion_matrix = ConfusionMatrix(y_e, y_p)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot()
plt.show()
