import sys
import operator
#from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k
    
    # Feel free to add parameters to this method
    def dfs(self,repr,visited=set()):
        freq_seqs = {}
        frequents_candidates = {k: v for k,
                                v in repr.items() if (len(v)) >= 2}
        for i in frequents_candidates.keys():
            visited.add(i)

        for i, j in list(frequents_candidates.items()):
            if type(i) == str:
                freq_seqs[tuple(i)] = len(
                    set([t[0] for t in j]))
            else:
                freq_seqs[tuple(i)] = len(
                    set([t[0] for t in j]))

        for x, val1 in sorted(list(frequents_candidates.items())):
            T = {}
            for y, val2 in sorted(list(frequents_candidates.items())):
                tmp=[]
                
                for t,n in val1:
                    for k,l in val2:
                        if t==k and l>n :
                            if tmp.count((k,l))==0:
                                tmp.append((k,l))
                if type(x) == str:
                    new_key = tuple((x, y))
                else:
                    new_key = tuple(list(i for i in x) + list(j for j in y))
                if len(tmp):
                    T[new_key] = tmp
            if len(T):
                freq_seqs.update(self.dfs(T, visited))
        return freq_seqs
    def get_transposed_data(self,dataset):
        """
        return a dictionnary where keys are items and values are sets
        that contains the id of transactions where the item appears.
        """
        new_data = {}
        for i in range(dataset.trans_num()):
            transaction = dataset.get_transaction(i)
            for item in transaction:
                if item not in new_data:
                    new_data[item] = set()
                    new_data[item].add(i)
                else:
                    new_data[item].add(i)
        return new_data
    def min_top_k(self):
        #c'est ici qu'on implemente le truc SPADE similaire à un DFS 
        pass
    
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
            classifier = DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = accuracy_score(m['test_labels'], predicted)
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
    return projected, cover 


def add_supports(d1, d2):
    result = {}
    for seq in set(d1.keys()).intersection(set(d2.keys())):
        val1 = d1.get(seq, 0)
        val2 = d2.get(seq, 0)
        result[seq] = val1 + val2
    return result

def most_frequent(my_dict,pos,neg,k):

        # trier le dictionnaire par valeur (fréquence)
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True)

    # filtrer les k éléments les plus fréquents
    most_frequent = sorted_dict[:k]

    # filtrer tous les autres éléments ayant le même support que le k-ème élément
    if k < len(sorted_dict):
        last_freq = most_frequent[-1][1]
        same_freq = [item for item in sorted_dict[k:] if item[1] == last_freq]
        most_frequent += same_freq

    # afficher les k éléments les plus fréquents et tous les autres éléments ayant le même support
    for key, value in most_frequent:
        print(list(key),pos[key], neg[key], value)


def main():
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    k = int(sys.argv[3])
    s = Spade(pos_filepath, neg_filepath, k)

    transac = spade_repr_from_transaction(s.pos_transactions)['repr']
    freq = spade_repr_from_transaction(s.pos_transactions)['covers']
    pos = s.dfs(transac)
    #print(pos)
    transac1 = spade_repr_from_transaction(s.neg_transactions)['repr']
    freq1 = spade_repr_from_transaction(s.neg_transactions)['covers']
    neg = s.dfs(transac1)


    result = add_supports(neg, pos)
    most_frequent(result, pos, neg, k)

if __name__ == '__main__':
    main()
