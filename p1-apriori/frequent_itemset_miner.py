import sys
import pandas as pd
import time
import matplotlib.pyplot as plt

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

__authors__ = "<Group 30, Patrick Tchoupe & Misselyn Lambert>"
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
    return new_data


def dfs(data, minFreq, n, visited=set()):
    #Rajouter le tri en ordre croissant ici
    frequents_candidates = {k: v for k,
                            v in data.items() if (len(v)/n) >= minFreq}
    for i in frequents_candidates.keys():
        visited.add(i)

    """ for i, j in list(frequents_candidates.items()):
        if type(i) == int:
            print(str([i]) + " (" + str(len(j)/n) + ")\n")
        else:
            print(str(list(i)) + " (" + str(len(j)/n) + ")\n")
 """
    for x, val1 in list(frequents_candidates.items()):
        T = {}
        for y, val2 in list(frequents_candidates.items()):
            if(x != y and len(val2) >= len(val1)):
                if type(x) == int:
                    new_key = tuple(sorted(list((x, y))))
                else:
                    new_key = tuple(
                        sorted(list(set(i for i in x).union(set(j for j in y)))))
                new_set = val1.intersection(val2)

                if ((len(new_set)/n) >= minFreq) and new_key not in visited:
                    T[new_key] = new_set
        if len(T):
            dfs(T, minFreq, n)


def support(data, candidates, n, minFreq):
    frequents_candidates = []

    for c in candidates:
        if type(c) == int:
            freq = len(data[c]) / n
            if freq >= minFreq:
                frequents_candidates.append(c)
                #print(str([c]) + " (" + str(freq) + ")")
                #file.write(str([c]) + " (" + str(freq) + ")\n")

        else:
            intersec = data[c[0]]
            for i in c[1:]:
                intersec = intersec.intersection(data[i])
            freq = len(intersec) / n
            if freq >= minFreq:
                frequents_candidates.append(c)
                #print(str(c) + " (" + str(freq) + ")")
                #file.write(str(c) + " (" + str(freq) + ")\n")
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
    dataset = Dataset(filepath)
    n_transactions = dataset.trans_num()
    transposed_data = get_transposed_data(dataset)
    candidates = transposed_data.keys()
    frequents_items = support(
        transposed_data, candidates, n_transactions, minFrequency)
    while len(frequents_items) > 2:
        candidates = generate_candidates(frequents_items)
        frequents_items = support(
            transposed_data, candidates, n_transactions, minFrequency)
    return frequents_items


def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    dataset = Dataset(filepath)
    n_transactions = dataset.trans_num()
    transposed_data = get_transposed_data(dataset)
    result = dfs(transposed_data, minFrequency, n_transactions)
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
    dataset = [
        "Datasets/mushroom/mushroom.dat",
        "Datasets/chess/chess.dat",
        "Datasets/accidents/accidents.dat",
    ]
    min_support = [1.0, 0.95, 0.9, 0.8, 0.85]
    frame = pd.DataFrame(
        columns=["Function", "File", "Support", "Execution time"])
    for supp in min_support:
        # Example of search
        start_timer = time.perf_counter()
        apriori("Datasets/retail/retail.dat", supp)
        end_timer = time.perf_counter()
        obj1 = {"Function": "Apriori", "File": "retail", "Support": supp,
                "Execution time": end_timer - start_timer}

        frame = frame.append(obj1, ignore_index=True)
    for supp in min_support:
        # Example of search
        start_timer = time.perf_counter()
        alternative_miner("Datasets/retail/retail.dat", supp)
        end_timer = time.perf_counter()
        obj1 = {"Function": "ECLAT", "File": "retail", "Support": supp,
                "Execution time": end_timer - start_timer}

        frame = frame.append(obj1, ignore_index=True)

    """ frame1  = pd.DataFrame(columns=["Function","File","Support","Execution time"])
    for ds in dataset :
        for supp in min_support :
                # Example of search
            start_timer = time.perf_counter()
            apriori(ds, supp)
            end_timer = time.perf_counter()
            obj1 = {"Function":"Apriori","File": ds,"Support": supp,"Execution time":end_timer - start_timer}

            frame1 = frame1.append(obj1, ignore_index=True)
    frame1.to_csv("result.csv") """

    """ frame2  = pd.DataFrame(columns=["Function","File","Support","Execution time"])
    for ds in dataset :
        for supp in min_support :
                # Example of search
            start_timer = time.perf_counter()
            alternative_miner(ds, supp)
            end_timer = time.perf_counter()
            obj2 = {"Function":"Apriori","File": ds,"Support": supp,"Execution time":end_timer - start_timer}

            frame2 = frame2.append(obj2, ignore_index=True)
    frame2.to_csv("result2.csv") """

    
    #frame = pd.read_csv("experiments.csv")
    #print(frame[frame["Function"]=="Apriori"])
    print(frame)
    pivot = pd.pivot_table(frame, values='Execution time', index=[
                           'File', 'Support'], columns=['Function'])

    # Créer un graphique en barres pour chaque fichier et support
    for file in frame['File'].unique():
        fig, ax = plt.subplots()
        pivot.loc[file].plot.bar(ax=ax)
        ax.set_xlabel('Minimum frequency')
        ax.set_ylabel('Execution time')
        ax.set_title(
            'Execution time for each function - ' + file)
        plt.show()
