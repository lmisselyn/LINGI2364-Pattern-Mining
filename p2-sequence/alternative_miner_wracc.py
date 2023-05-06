import numpy as np
import xgboost
from sklearn import tree, metrics
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from helper import *


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k

    # Feel free to add parameters to this method
    def min_top_k(self):
        frequents = []
        N = len(self.neg_transactions)
        P = len(self.pos_transactions)
        pos = spade_repr_from_transaction(self.pos_transactions)
        neg = spade_repr_from_transaction(self.neg_transactions)
        # Compute the support of different symbols and sort them
        symbol_support = get_symbols_support(pos['covers'], neg['covers'])
        sorted_symbols = sorted(
            symbol_support, key=lambda i: -symbol_support[i][2])

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

            pos_positions = find_sub_sequence(
                item[0], item[1], self.pos_transactions)

            neg_positions = find_sub_sequence(
                item[0], item[2], self.neg_transactions)
            pos_supp = get_support_from_pos(pos_positions)
            neg_supp = get_support_from_pos(neg_positions)

            # Keep track of the minimum score during search

            if neg_supp + pos_supp >= min_score:
                strItem = item_str(item[0])
                symbol_support[strItem] = [
                    pos_supp, neg_supp, pos_supp + neg_supp]
                for symb in best_symbols:
                    new_symbol = item[0].copy()
                    new_symbol.append(symb)
                    frequents.append(
                        (new_symbol, pos_positions, neg_positions))

        tmp = {}
        for key, value in symbol_support.items():
            i = value[0]
            j = value[1]
            tmp[key] = [i, j, round(wracc(P, N, i, j), 5)]

        sorted_result = sorted(tmp.items(), key=lambda i: -i[1][2])

        return sorted_result[0]

    def get_feature_matrices(self):
        return {
            'train_matrix': [],
            'test_matrix': [],
            'train_labels': [],
            'test_labels': [],
        }

    def cross_validation(self, nfolds):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set = {i for i in range(len(self.pos_transactions)) if
                             i < fold * pos_fold_size or i >= (fold + 1) * pos_fold_size}
            neg_train_set = {i for i in range(len(self.neg_transactions)) if
                             i < fold * neg_fold_size or i >= (fold + 1) * neg_fold_size}

            self.min_top_k()

            m = self.get_feature_matrices()
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')

    def alternative_mine_top_k(self):
        #fix check_presence
        n_iter = self.k
        self.k = 1
        result = []
        for i in range(n_iter):
            pattern = self.min_top_k()[0]
            if pattern == False:
                return result
            if type(pattern) == type("ok"):
                pattern = [pattern]
            result.append(pattern)
            present_pos = check_presence(pattern, self.pos_transactions)
            present_neg = check_presence(pattern, self.neg_transactions)

            new_pos_transactions = []
            new_neg_transactions = []

            for j in range(len(present_pos)):
                if present_pos[j] == 0:
                    new_pos_transactions.append(self.pos_transactions[j])
            for j in range(len(present_neg)):
                if present_neg[j] == 0:
                    new_neg_transactions.append(self.neg_transactions[j])

            self.pos_transactions = new_pos_transactions
            self.neg_transactions = new_neg_transactions
        return result



def cross_val(n, X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    res = []
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        model = XGBClassifier(booster='gbtree')
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        res.append(metrics.accuracy_score(y_test, y_predict))
    print(res)
    return res




if __name__ == '__main__':

    print(round(np.mean([0.8345323741007195, 0.8705035971223022, 0.8273381294964028, 0.9064748201438849, 0.8633093525179856]), 3))
    #pos_filepath = sys.argv[1]
    #neg_filepath = sys.argv[2]
    #k = int(sys.argv[3])
    #pos_filepath = 'datasets/Reuters_small/positive_earn_small.txt'
    #neg_filepath = 'datasets/Reuters_small/negative_acq_small.txt'
    pos_filepath = 'datasets/Protein/positive_SRC1521.txt'
    neg_filepath = 'datasets/Protein/negative_PKA_group15.txt'
    #pos_filepath = 'datasets/Test/positive.txt'
    #neg_filepath = 'datasets/Test/negative.txt'


    accuracies = []
    for i in range(1, 11):
        k = i
        s = Spade(pos_filepath, neg_filepath, k)
        values = []
        result = []

        top_patterns = s.alternative_mine_top_k()
        print(top_patterns)

        for p in top_patterns:
            values.append(p[0])
            result.append(p[0].split(', '))

        values.append("Class")

        t = Spade(pos_filepath, neg_filepath, k)
        rows = build_repr(result, t.pos_transactions, t.neg_transactions)

        frame = pd.DataFrame(rows, columns=values)


        #frame = frame.append(pd.DataFrame(
         #   rows, columns=frame.columns), ignore_index=True)

        X = frame.drop('Class', axis=1)
        y = frame['Class']

        accuracies.append(round(np.mean(cross_val(5, X, y)), 3))
    print(accuracies)
