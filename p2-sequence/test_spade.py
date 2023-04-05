#!/usr/bin/env python3
import sys
import time
"""
Frequent Sequence Miner for the project 2 of the LINGI2364 course
__authors__ = "Group 6, Thomas Robert & Luc Cleenewerk"
"""

class Dataset:
    """Utility class to manage a dataset stored in a external file."""
    def __init__(self, pos_filepath, neg_filepath):
        """reads the dataset file and initializes files"""
        self.transactions_pos = list()
        self.transactions_neg = list()
        self.verti_pos = dict()
        self.verti_neg = dict()
        try:
            self.init_transaction(pos_filepath, self.transactions_pos)
            self.init_transaction(neg_filepath, self.transactions_neg)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def init_transaction(self, filepath, transaction_class):
        """Init the transactions of the file"""
        lines = [line.strip() for line in open(filepath, "r")]
        transaction = []
        for line in lines:
            if not line:
                transaction_class.append(transaction)
                transaction = []
            else:
                symbol, position = line.split(" ")
                transaction.append(symbol)

    def verti_repres(self):
        """
        Returns the vertical representation of the Dataset
        A tuple of the vertical representation of the positive file and same for the negative file
        """
        for i in range(len(self.transactions_pos)):
            trans = self.transactions_pos[i]
            for j in range(len(trans)):
                if trans[j] not in self.verti_pos.keys():
                    self.verti_pos[trans[j]] = []
                self.verti_pos[trans[j]].append((i, j+1))
        for i in range(len(self.transactions_neg)):
            trans = self.transactions_neg[i]
            for j in range(len(trans)):
                if trans[j] not in self.verti_neg.keys():
                    self.verti_neg[trans[j]] = []
                self.verti_neg[trans[j]].append((i, j+1))
        return self.verti_pos, self.verti_neg

def semi_verti_repres(verti1, verti2):
    """
    verti1&2 are array of position (i, j)
    Compute the resulting array of position (i, j) by taking verti2 and
    see if an element of verti1 was seen before on the same transaction
    """
    v1 = 0
    ret = []
    # Test if one of the two is not represented, the computation end directly
    if type(verti1) == type(None) or type(verti2) == type(None) or len(verti1) == 0 or len(verti2) == 0: return ret
    for pos in verti2:
        while v1 < len(verti1)-1 and verti1[v1][0] < pos[0]:
            v1 += 1
        if verti1[v1][0] == pos[0] and verti1[v1][1] < pos[1]:
            ret.append(pos)
    return ret

def support(verti):
    """
    Compute the support of an array of position (i, j)
    The support is the number of different transactions where it appears.
    """
    ret = 0
    seen = set()
    for i in verti:
        if i[0] not in seen:
            ret+=1
            seen.add(i[0])
    return ret

def array_printer(String):
    """
    Transform a String into a formatted string array
    Ex: "A,B,C" -> "[A, B, C]"
    """
    ret = "["
    for symbol in String.split(","):
        ret += symbol + ", "
    ret = ret[:-2]
    ret += "]"
    return ret

def spade(pos, neg, k):
    """
    Spade algorithm: Dfs using the vertical representation of transactions.
    pos: vertical representation of the positive file
    neg: vertical representation of the negative file
    k: the number of best frequent sequence to print.
    """
    t1 = time.time()
    best = dict()   # Dictionnary of key sequence: value [posi support, nega support, total support]
    symbols = []
    stack = []
    # 1st layer of the tree computation
    max = 0  # Threshold computation
    all = 0  # Threshold computation
    for classe in [pos, neg]:
        max_temp = 0
        for symbol in classe.keys():
            support_symbol = support(classe.get(symbol))

            # Threshold computation
            all += support_symbol
            if support_symbol > max_temp:
                max_temp = support_symbol

            # Setup best values for 1st layer
            if symbol not in best:
                best[symbol] = [0, 0, 0]
                symbols.append(symbol)
            if classe==pos:
                best[symbol][0] = support_symbol
                best[symbol][2] += support_symbol
            else:
                best[symbol][1] = support_symbol
                best[symbol][2] += support_symbol
        max += max_temp
    threshold = round(max/k) # Threshold computation
    # Cancel base symbols if they are not enough present regarding the threshold

    print(threshold)
    best_symbols = []
    for sy in symbols:
        if best[sy][2] > threshold:
            best_symbols.append(sy)
    symbols = best_symbols
    # Initiate next values on the stack
    for sy in symbols:
        pos_sy = pos.get(sy)
        neg_sy = neg.get(sy)
        for sy2 in symbols:
            stack.append([sy+","+sy2, pos_sy, neg_sy])
    print(stack)

    # All other layer of the tree using the stack
    while (len(stack) > 0):
        to_check = stack[-1]  # Get top of the stack
        print(to_check)
        stack = stack[:-1]    # Actualize the stack
        pos_semi_verti = semi_verti_repres(to_check[1], pos.get(to_check[0].split(",")[-1], []))
        neg_semi_verti = semi_verti_repres(to_check[2], neg.get(to_check[0].split(",")[-1], []))
        print(pos_semi_verti)
        print(neg_semi_verti)
        for classe in [pos, neg]:
            if to_check[0] not in best:
                best[to_check[0]] = [0, 0, 0]
            if classe==pos:
                best[to_check[0]][0] = support(pos_semi_verti)
                best[to_check[0]][2] += support(pos_semi_verti)
            else:
                best[to_check[0]][1] = support(neg_semi_verti)
                best[to_check[0]][2] += support(neg_semi_verti)
        # Launch next values on the stack only if support is higher than the treshold
        if best[to_check[0]][2] > threshold:
            for symbol in symbols:
                stack.append([to_check[0]+","+symbol, pos_semi_verti, neg_semi_verti])
    # Sort the sequences by total support
    best_sorted = sorted(best.items(), key=lambda i: -i[1][2])
    old_support = 0
    i = 0
    while (i < len(best_sorted)):
        x = best_sorted[i]
        # If same support as previous one, it doesn't decrease k
        if not (old_support == x[1][2]):
            k-=1
            # Cancel the while loop when we printed enough
            if k < 0:
                # t4 = time.time()
                # print("After output solution", t4-t1)
                return best_sorted
            old_support = x[1][2]
        #print(array_printer(x[0]) + " " + str(x[1][0]) + " " + str(x[1][1]) + " " + str(x[1][2]))
        i+=1
    return best_sorted

def main():
    # /bin/python3 /home/path/to/p2.py Test/positive.txt Test/negative.txt 5
    # pos_filepath = sys.argv[1] # filepath to positive class file
    # neg_filepath = sys.argv[2] # filepath to negative class file
    # k = int(sys.argv[3])
    pos_filepath = "datasets/Test/positive.txt"
    neg_filepath = "datasets/Test/negative.txt"
    k = 6

    data = Dataset(pos_filepath, neg_filepath)
    verti = data.verti_repres()
    spade(verti[0], verti[1], k)

if __name__ == "__main__":
    main()