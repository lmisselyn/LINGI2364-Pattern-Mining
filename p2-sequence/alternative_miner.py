from sklearn import tree, metrics
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

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
        #best_symbols = sorted_symbols[:self.k]
        #min_score = symbol_support[best_symbols[-1]][2]

        min_score = symbol_support[sorted_symbols[0]][2]/self.k
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


            # Keep track of the minimum score during search


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


def get_transactions(filepath):
    """
    Return the list of transactions ine the file
    """
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert (int(element[1]) - 1 == len(transactions[-1]))
                transactions[-1].append(element[0])
            else:
                new_transaction = True
    return transactions


def spade_repr_from_transaction(transactions):
    spade_repr = {}
    covers = {}
    for tid, transaction in enumerate(transactions):
        for i, item in enumerate(transaction):
            try:
                covers[item].add(tid)
            except KeyError:
                covers[item] = {tid}
            try:
                spade_repr[item].append((tid, i))
            except KeyError:
                spade_repr[item] = [(tid, i)]
    return {'repr': spade_repr, 'covers': covers}
    # return projected, cover


def get_symbols_support(pos_cover, neg_cover):
    """
    Compute the total support of each symbol
    appearing in pos and neg files.
    """
    symbols_support = {}
    for symbol in pos_cover.keys():
        supp = len(pos_cover[symbol])
        symbols_support[symbol] = [supp, 0, supp]
    for symbol in neg_cover.keys():
        supp = len(neg_cover[symbol])
        if symbol in symbols_support:
            symbols_support[symbol][1] = supp
            symbols_support[symbol][2] = supp + symbols_support[symbol][0]
        else:
            symbols_support[symbol] = [0, supp, supp]

    return symbols_support


def get_support_from_pos(positions):
    """
    Return the support of a pattern based on its positions
    int the transactions. Max 1 support by transaction.
    """
    transaction = []
    supp = 0
    for pos in positions:
        if pos[0] not in transaction:
            supp += 1
            transaction.append(pos[0])
    return supp


def find_sub_sequence(item, positions, transactions):
    """
    Return the positions in the transactions where the
    pattern (item) appears.
    """
    new_positions = []
    for pos in positions:
        seq = transactions[pos[0]]
        for i in range(pos[1] + 1, len(seq)):
            new_pos = (pos[0], i)
            if seq[i] == item[-1] and new_pos not in new_positions:
                new_positions.append(new_pos)
    return new_positions


def item_str(item):
    s = ""
    for i in range(len(item)):
        s += str(item[i])
        if i != len(item) - 1:
            s += ', '
    return s


def print_ressult2(res):
    for r in res:
        symb = r[0]
        print('['+ symb + ']' + ' ' + str(r[1][0]) + ' ' + str(r[1][1]) + ' ' + str(r[1][2]))


def print_result(res):
    for r in res:
        symb = '['
        for i in range(len(r[0])):
            if r[0][i] != ',':
                if i != len(r[0]) - 1:
                    symb += r[0][i] + ', '
                else:
                    symb += r[0][i] + ']'
        print(symb + ' ' + str(r[1][0]) + ' ' + str(r[1][1]) + ' ' + str(r[1][2]))


def check_presence(pattern, transactions):
    """
    Return a list for the presence of the pattern in each transaction
    1 - pattern present in the transaction
    0 - pattern not present
    """
    pres=[]
    for transaction in transactions:
        i = 0
        for element in pattern:
            if element not in transaction[i:]:
                # not present
                pres.append(0)
                break
            # do the search from index i
            i = transaction.index(element, i) + 1
        else:
            # the pattern is present
            pres.append(1)
    return pres


def build_repr(result, pos, neg):
    """
    Return the representation needed by sklearn models
    """
    rows1 = []
    for pat in result:
        tmp = check_presence(pat, pos)
        rows1.append(tmp)

    rows1 = [list(col) for col in zip(*rows1)]
    for l in rows1:
        l.append("P")

    rows2 = []
    for pat in result:
        tmp = check_presence(pat, neg)
        rows2.append(tmp)

    rows2 = [list(col) for col in zip(*rows2)]
    for l in rows2:
        l.append("N")

    rows = rows1 + rows2

    return rows


def cross_val(n, X, y):
    model = tree.DecisionTreeClassifier(random_state=1)
    scores = cross_val_score(model, X=X, y=y, cv=n, scoring='accuracy')
    print(scores)

if __name__ == '__main__':
    #pos_filepath = sys.argv[1]
    #neg_filepath = sys.argv[2]
    #k = int(sys.argv[3])
    pos_filepath = 'datasets/Reuters_small/positive_earn_small.txt'
    neg_filepath = 'datasets/Reuters_small/negative_acq_small.txt'
    #pos_filepath = 'datasets/Test/positive.txt'
    #neg_filepath = 'datasets/Test/negative.txt'
    k = int(7)
    s = Spade(pos_filepath, neg_filepath, k)
    patterns = s.alternative_mine_top_k()

    values = []
    result = []

    for i in patterns:
        tmp = []
        values.append(i[0])
        for j in i[0]:
            if str(j).isalpha():
                tmp.append(str(j))
        result.append(tmp)

    values.append("Class")

    rows = build_repr(result, s.pos_transactions, s.neg_transactions)

    frame = pd.DataFrame(columns=values)

    frame = frame.append(pd.DataFrame(
        rows, columns=frame.columns), ignore_index=True)

    X = frame.drop('Class', axis=1)
    y = frame['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    m = {
        'train_matrix': X_train,
        'test_matrix': X_test,
        'train_labels': y_train,
        'test_labels': y_test,
    }
    cross_val(5, X, y)