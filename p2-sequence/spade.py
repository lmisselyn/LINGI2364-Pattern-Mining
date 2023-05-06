import math
import sys
from helper import *
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k

    # Feel free to add parameters to this method
    def min_top_k(self):
        frequents = []
        pos = spade_repr_from_transaction(self.pos_transactions)
        neg = spade_repr_from_transaction(self.neg_transactions)
        # Compute the support of different symbols and sort them
        symbol_support = get_symbols_support(pos['covers'], neg['covers'])
        sorted_symbols = sorted(symbol_support, key=lambda i: -symbol_support[i][2])

        # Prune base symbols
        min_score = symbol_support[sorted_symbols[0]][2] / self.k
        index = 0
        for s in sorted_symbols:
            if symbol_support[s][2] < min_score:
                break
            index += 1
        best_symbols = sorted_symbols[:index]

        # Build patterns of size 2
        for symb in best_symbols:
            for symb2 in best_symbols:
                frequents.append(([symb, symb2],
                                  pos['repr'][symb] if symb in pos['repr'] else [],
                                  neg['repr'][symb] if symb in neg['repr'] else []))

        # DFS
        while len(frequents) > 0:
            item = frequents.pop()
            pos_positions = find_sub_sequence(item[0], item[1], self.pos_transactions)
            neg_positions = find_sub_sequence(item[0], item[2], self.neg_transactions)
            pos_supp = get_support_from_pos(pos_positions)
            neg_supp = get_support_from_pos(neg_positions)

            if neg_supp + pos_supp >= min_score:
                strItem = item_str(item[0])
                symbol_support[strItem] = [pos_supp, neg_supp, pos_supp + neg_supp]
                for symb in best_symbols:
                    new_symbol = item[0].copy()
                    new_symbol.append(symb)
                    frequents.append((new_symbol, pos_positions, neg_positions))
        sorted_result = sorted(symbol_support.items(), key=lambda i: -i[1][2])

        final_result = []
        previous_support = 0
        # Return top k patterns
        # All patterns with the same score worth for 1 k
        # print(sorted_result)
        size = self.k
        for item in sorted_result:

            if size != 0:
                if item[1][2] != previous_support:
                    if previous_support != 0:
                        size -= 1
                    if size == 0:
                        return final_result
                    previous_support = item[1][2]
                    final_result.append(item)
                else:
                    final_result.append(item)
            else:
                return final_result
        return final_result

    def get_feature_matrices(self):
        return {
            'train_matrix': [],
            'test_matrix': [],
            'train_labels': [],
            'test_labels': [],
        }

    def cross_validation(self, nfolds, m):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set = {i for i in range(len(self.pos_transactions)) if
                             i < fold * pos_fold_size or i >= (fold + 1) * pos_fold_size}
            neg_train_set = {i for i in range(len(self.neg_transactions)) if
                             i < fold * neg_fold_size or i >= (fold + 1) * neg_fold_size}

            # self.min_top_k()

            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')




def cross_val(n, X, y):
    model = tree.DecisionTreeClassifier(random_state=1)
    scores = cross_val_score(model, X=X, y=y, cv=n, scoring='accuracy')
    print(scores)
    return scores


if __name__ == '__main__':
    # pos_filepath = sys.argv[1]
    # neg_filepath = sys.argv[2]
    # k = int(sys.argv[3])
    pos_filepath = 'datasets/Reuters_small/positive_earn_small.txt'
    neg_filepath = 'datasets/Reuters_small/negative_acq_small.txt'
    #pos_filepath = 'datasets/Protein/positive_SRC1521.txt'
    #neg_filepath = 'datasets/Protein/negative_PKA_group15.txt'
    #pos_filepath = 'datasets/Test/positive.txt'
    #neg_filepath = 'datasets/Test/negative.txt'

    accuracies = []
    for i in range(1, 11):
        k = i
        s = Spade(pos_filepath, neg_filepath, k)
        values = []
        result = []

        top_patterns = s.min_top_k()

        for j in top_patterns:
            values.append(j[0])
            result.append(j[0].split(', '))
        print(result)
        values.append("Class")

        rows = build_repr(result, s.pos_transactions, s.neg_transactions)

        frame = pd.DataFrame(columns=values)

        frame = frame.append(pd.DataFrame(
            rows, columns=frame.columns), ignore_index=True)

        X = frame.drop('Class', axis=1)
        y = frame['Class']

        accuracies.append(round(np.mean(cross_val(5, X, y)), 3))
    print(accuracies)
