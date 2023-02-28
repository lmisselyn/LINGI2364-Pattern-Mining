import sys
from itertools import combinations

"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<write here your group, first name(s) and last name(s)>"
"""


class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
                transaction = list(map(int, line.split(" ")))
                self.transactions.append(transaction)
                for item in transaction:
                    self.items.add(item)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self.transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]




def get_transposed_data(dataset):
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
    new_data =  dict(sorted(new_data.items(), key=lambda item: len(item[1])))
    return new_data


def dfs(data,minFreq,n,file):

    frequents_candidates = {k: v for k, v in data.items() if (len(v)/n) >= minFreq}
    
    for i, j in list(frequents_candidates.items()):
        if type(i) == int :
            file.write(str([i]) + " (" + str(len(j)/n) + ")\n")
        else :
            file.write(str(list(i)) + " (" + str(len(j)/n) + ")\n")
        

    for x, val1 in list(frequents_candidates.items()):
        T = {}
        for y, val2 in list(frequents_candidates.items()):
            if(x!=y and len(val2)>len(val1)) :
                if type(x) == int :
                    new_key = (x,y)
                else:
                    new_key = tuple(set(i for i in x).union(set(j for j in y)))
                new_set = val1.intersection(val2)                
                
                if ( (len(new_set)/n) >= minFreq) :
                    print(str(list(new_key)) + " (" + str(len(new_set)/n) + ")")
                    T[new_key] = new_set
        if len(T):
            dfs(T,minFreq,n,file)



def support(data, candidates, n, minFreq, file):
    frequents_candidates = []

    for c in candidates:
        if type(c) == int:
            freq = len(data[c]) / n
            if freq >= minFreq:
                frequents_candidates.append(c)
                print(str([c]) + " (" + str(freq) + ")")
                file.write(str([c]) + " (" + str(freq) + ")\n")

        else:
            intersec = data[c[0]]
            for i in c[1:]:
                intersec = intersec.intersection(data[i])
            freq = len(intersec) / n
            if freq >= minFreq:
                frequents_candidates.append(c)
                print(str(c) + " (" + str(freq) + ")")
                file.write(str(c) + " (" + str(freq) + ")\n")
    return frequents_candidates


def generate_candidates(itemsets):
    candidates = []
    for itemset in itemsets:
        for itemset2 in itemsets:
            if itemset != itemset2:
                if type(itemset) == int:
                    new_cand = sorted([itemset, itemset2])
                    if new_cand not in candidates:
                        candidates.append(new_cand)

                else:
                    new_cand = itemset.copy()
                    if itemset2[-1] not in itemset:
                        new_cand.append(itemset2[-1])
                        new_cand.sort()
                        if new_cand not in candidates:
                            candidates.append(new_cand)
    return list(candidates)


def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    f = open("MySols/sol.dat", "w")
    dataset = Dataset(filepath)
    n_transactions = dataset.trans_num()
    transposed_data = get_transposed_data(dataset)
    candidates = transposed_data.keys()
    frequents_items = support(transposed_data, candidates, n_transactions, minFrequency, f)
    while len(frequents_items) > 2:
        candidates = generate_candidates(frequents_items)
        frequents_items = support(transposed_data, candidates, n_transactions, minFrequency, f)
    f.close()
    return frequents_items

def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    f = open("MySols/sol.dat", "w")
    dataset = Dataset(filepath)
    n_transactions = dataset.trans_num()
    transposed_data = get_transposed_data(dataset)
    result = dfs(transposed_data,minFrequency,n_transactions,f)
    f.close()
    return result


'''
Helper :
dataset : is a dictionnary where keys are items and values are arrays that contains the id of transactions where the 
item appears. 

idea : take itemset of size 1, compute support and keep itemsets with a freq > minFrequency.
generate next candidates (itemsets of size 2) with the previous itemsets
compute support on these new candidates and so on...
'''

if __name__ == '__main__':
   print(alternative_miner("Datasets/chess/chess.dat", 0.8))

