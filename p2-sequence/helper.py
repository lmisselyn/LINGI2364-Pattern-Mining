
def wracc(P, N, px, nx):
    return (P / (P + N)) * (N / (P + N)) * ((px / P) - (nx / N))


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


def check_presence(pattern, transactions):
    """
    Return a list for the presence of the pattern in each transaction
    1 - pattern present in the transaction
    0 - pattern not present
    """
    pres = []
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


def item_str(item):
    s = ""
    for i in range(len(item)):
        s += str(item[i])
        if i != len(item) - 1:
            s += ', '

    return s


def print_result2(res):
    for r in res:
        symb = r[0]
        print('[' + symb + ']' + ' ' + str(r[1][0]) +
              ' ' + str(r[1][1]) + ' ' + str(r[1][2]))


def print_result(res):
    for r in res:
        symb = '['
        for i in range(len(r[0])):
            if r[0][i] != ',':
                if i != len(r[0]) - 1:
                    symb += r[0][i] + ', '
                else:
                    symb += r[0][i] + ']'
        print(symb + ' ' + str(r[1][0]) + ' ' +
              str(r[1][1]) + ' ' + str(r[1][2]))


def build_repr(result, pos, neg):
    """
    Return the representation needed by sklearn models
    """
    rows1 = []
    for pat in result:
        if type(pat) == type("ok"):
            pat = [pat]
        tmp = check_presence(pat, pos)
        rows1.append(tmp)

    rows1 = [list(col) for col in zip(*rows1)]
    for l in rows1:
        l.append("P")

    rows2 = []
    for pat in result:
        if type(pat) == type("ok"):
            pat = [pat]
        tmp = check_presence(pat, neg)
        rows2.append(tmp)

    rows2 = [list(col) for col in zip(*rows2)]
    for l in rows2:
        l.append("N")

    rows = rows1 + rows2

    return rows