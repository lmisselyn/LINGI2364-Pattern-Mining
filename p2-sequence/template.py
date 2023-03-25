import sys
from sklearn import tree, metrics


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k
        
    # Feel free to add parameters to this method
    def min_top_k(self):
        pos_cover = spade_repr_from_transaction(self.pos_transactions)['covers']
        neg_cover = spade_repr_from_transaction(self.neg_transactions)['covers']

        symbol_support = get_symbols_support(pos_cover, neg_cover)



    
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
            pos_train_set = {i for i in range(len(self.pos_transactions)) if i < fold*pos_fold_size or i >= (fold+1)*pos_fold_size}
            neg_train_set = {i for i in range(len(self.neg_transactions)) if i < fold*neg_fold_size or i >= (fold+1)*neg_fold_size}

            self.mine_top_k()
            
            m = self.get_feature_matrices()
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')

def get_transactions(filepath):
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert(int(element[1]) - 1 == len(transactions[-1]))
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
    #return projected, cover


def get_symbols_support(pos_cover, neg_cover):
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

if __name__ == '__main__':
    #pos_filepath = sys.argv[1]
    #neg_filepath = sys.argv[2]
    #k = int(sys.argv[3])
    pos_filepath = 'datasets/Reuters_small/positive_earn_small.txt'
    neg_filepath = 'datasets/Reuters_small/negative_acq_small.txt'
    k = int(3)
    s = Spade(pos_filepath, neg_filepath, k)
    s.min_top_k()