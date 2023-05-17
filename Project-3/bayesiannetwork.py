import math
import random
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



# A class for representing the Conditional Probability Tables of a variable
class CPT:

    def __init__(self, head, parents):
        self.head = head  # The variable this CPT belongs to (object)
        self.parents = parents  # Parent variables (objects), in order
        self.entries = {}
        # Entries in the CPT. The key of the dictionnary is an
        # assignment to the parents; the associated value is a dictionnary
        # itself, reflecting one row in the CPT.
        # For a variable that has no parents, the key is the empty tuple.

    # String representation of the CPT according to the BIF format
    def __str__(self):
        comma = ", "
        if len(self.parents) == 0:
            return f"probability ( {self.head.name} ) {{" + "\n" \
                                                            f"  table {comma.join(map(str, self.entries[tuple()].values()))};" + "\n" \
                                                                                                                                 f"}}" + "\n"
        else:
            return f"probability ( {self.head.name} | {comma.join([p.name for p in self.parents])} ) {{" + "\n" + \
                   "\n".join([ \
                       f"  ({comma.join(names)}) {comma.join(map(str, values.values()))};" \
                       for names, values in self.entries.items() \
                       ]) + "\n}\n"

        # A class for representing a variable


class Variable:

    def __init__(self, name, values):
        self.name = name  # Name of the variable
        self.values = values
        # The domain of the variable: names of the values
        self.cpt = None  # No CPT initially

    # String representation of the variable according to the BIF format
    def __str__(self):
        comma = ", "
        return f"variable {self.name} {{" + "\n" \
               + f"  type discrete [ {len(self.values)} ] {{ {(comma.join(self.values))} }};" + "\n" \
               + f"}}" + "\n"


class BayesianNetwork:

    # Method for reading a Bayesian Network from a BIF file;
    # fills a dictionary 'variables' with variable names mapped to Variable
    # objects having CPT objects.
    def __init__(self, input_file, datafile):

        with open(input_file) as f:
            lines = f.readlines()

        self.input_file = input_file
        self.data_file = datafile
        self.assignements = self.get_assignments_from_file()
        self.variables = {}
        # The dictionary of variables, allowing quick lookup from a variable
        # name.
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace('/', '-')

        if len(lines) != 0:
            # Parsing all the variable definitions
            i = 0
            while not lines[i].startswith("probability"):
                if lines[i].startswith("variable"):
                    variable_name = lines[i].rstrip().split(' ')[1]
                    i += 1
                    variable_def = lines[i].rstrip().split(' ')
                    # only discrete BN are supported
                    assert (variable_def[1] == 'discrete')
                    variable_values = [x for x in variable_def[6:-1]]
                    for j in range(len(variable_values)):
                        variable_values[j] = re.sub('\(|\)|,', '', variable_values[j])
                    variable = Variable(variable_name, variable_values)
                    self.variables[variable_name] = variable
                i += 1

            # Parsing all the CPT definitions
            while i < len(lines):
                split = lines[i].split(' ')
                target_variable_name = split[2]
                variable = self.variables[target_variable_name]

                parents = [self.variables[x.rstrip().lstrip().replace(',', '')] for x in split[4:-2]]

                assert (variable.name == split[2])

                cpt = CPT(variable, parents)
                i += 1

                nb_lines = 1
                for p in parents:
                    nb_lines *= len(p.values)
                for lid in range(nb_lines):
                    cpt_line = lines[i].split(' ')
                    # parent_values = [parents[j].values[re.sub('\(|\)|,', '', cpt_line[j])] for j in range(len(parents))]
                    parent_values = tuple([re.sub('\(|\)|,', '', cpt_line[j]) for j in range(len(parents))])
                    probabilities = re.findall("\d\.\d+(?:e-\d\d)?", lines[i])
                    cpt.entries[parent_values] = {v: float(p) for v, p in zip(variable.values, probabilities)}
                    i += 1
                variable.cpt = cpt
                i += 1

    # Method for writing a Bayesian Network to an output file
    def write(self, filename):
        with open(filename, "w") as file:
            for var in self.variables.values():
                file.write(str(var))
            for var in self.variables.values():
                file.write(str(var.cpt))

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and x an assignment to its parents in the network, specified
    # in the correct order of parents.
    def P_Yisy_given_parents_x(self, Y, y, x=tuple()):
        return self.variables[Y].cpt.entries[x][y]

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and pa a dictionnary of assignments to the parents,
    # with for every parent variable its associated assignment.
    def P_Yisy_given_parents(self, Y, y, pa={}):
        x = tuple([pa[parent.name] for parent in self.variables[Y].cpt.parents])
        return self.P_Yisy_given_parents_x(Y, y, x)

    def P_Yisy(self, Y, y):
        P = 1
        CPT = self.variables[Y].cpt
        for entry in CPT.entries.keys():
            P = P * CPT.entries[entry][y]
        return P

    def P_transaction(self, transaction):
        """
        :param transaction: assignment to every variable of the network
        :return: probability of such an assignment
        """
        p = 1
        for var in transaction.keys():
            var_parents = {}
            for pa in self.variables[var].cpt.parents:
                var_parents[pa.name] = str(transaction[pa.name])
            p_Xisx = self.P_Yisy_given_parents(var, str(transaction[var]), var_parents)
            p = p * p_Xisx
        return p

    def get_distrib_givenX(self, Y, x={}):
        """
        :param Y: one variable
        :param x: assignments for all other variables
        :return: distribution probability for all possibles assignments of Y
        """
        res_distrib = {}
        CPT = self.variables[Y].cpt
        # Joint distribution
        for y in self.variables[Y].values:
            transac = x.copy()
            transac[Y] = y
            p_numerator = self.P_transaction(transac)

            p_denominator = 0
            for val in self.variables[Y].values:
                transac = x.copy()
                transac[Y] = val
                p_denominator += self.P_transaction(transac)
            res_distrib[y] = p_numerator / p_denominator
        return res_distrib

    def get_distrib_givenX_double(self, Y, x={}):
        """
        :param Y: two variables
        :param x: assignments for all other variables
        :return: distribution probability for all possibles assignments of Y
        """
        res_distrib = {}
        new_Y = []
        for v in Y:
            new_Y.append(self.variables[v])
        combinaisons = self.get_combinations(new_Y)
        # denominator
        p_denominator = 0
        for tup in combinaisons:
            transac = x.copy()
            transac[Y[0]] = tup[0]
            transac[Y[1]] = tup[1]
            p_denominator += self.P_transaction(transac)
        # numerator
        for tup in combinaisons:
            transac = x.copy()
            transac[Y[0]] = tup[0]
            transac[Y[1]] = tup[1]
            p_numerator = self.P_transaction(transac)
            res_distrib[(tup[0], tup[1])] = p_numerator / p_denominator
        return res_distrib

    def get_combinations(self, variables):
        """
        :param variables:
        :return: All possibles assignments for these variables
        """
        arr = []
        for v in variables:
            arr.append(self.variables[v.name].values)
        combinations = list(itertools.product(*arr))
        return combinations

    def get_assignments_from_file(self):
        """
        :return: each row of the datafile as a dictionary
        """
        df = pd.read_csv(self.data_file)
        return df.to_dict(orient='records')
    
    def get_alternative_score(self):
        """
        :return: Compute de score of the network (sum of loglikelihood - number of parameters)
        """
        score = 0
        for assignment in self.assignements:
            proba = self.P_transaction(assignment)
            score += math.log(proba)
        param = 0
        for v in self.variables:
            if len(self.variables[v].cpt.parents):
                param += 4
            else:
                param += 2
        return score - param

    def get_score(self):
        """
        :return: Compute the score of the network (sum of loglikelihood)
        """
        score = 0
        for assignment in self.assignements:
            proba = self.P_transaction(assignment)
            score += math.log(proba)
        return score

    def param_learning(self, variable, network_name):
        """
        Compute the conditional probability table for the given variable
        based on the bayesian network
        :param variable: name of the variable of which we compute CPT
        :param network_name: name of the bayesian network
        """
        new_entry = {}
        df = pd.read_csv(network_name)
        parents = self.variables[variable].cpt.parents
        combinations = self.get_combinations(parents)

        for assignment in combinations:
            data = df.copy()
            for i in range(len(parents)):
                data = data.loc[df[parents[i].name] == int(assignment[i])]
            count = {}
            c = len(self.variables[variable].values)

            for i in self.variables[variable].values:
                count[str(i)] = 0
            for i in data[variable].values:
                count[str(i)] = count[str(i)] + 1
            for k in count.keys():
                count[k] = round((count[k] + 0.02) / (len(data[variable].values) + c * 0.02), 4)
            new_entry[assignment] = count
        self.variables[variable].cpt.entries = new_entry


def structure_init(network_name, filename):
    """
    :param network_name: name of the initiated network
    :param filename: Name of the csv file containing data
    :return: Initiate a Bayesian network with independent nodes
    """
    f = open("networks/" + network_name, 'w')
    f.close()
    new_bn = BayesianNetwork("networks/" + network_name, filename)
    df = pd.read_csv(filename)
    variables = df.columns
    for v in variables:
        uniques = df[v].unique()
        uniques.sort()

        count = {}
        for i in df[v].unique():
            count[i] = 0
        for i in df[v].values:
            count[i] = count[i] + 1
        for k in count.keys():
            count[k] = count[k] / len(df[v].values)
        new_var = Variable(v, [str(v) for v in uniques])
        new_cpt = CPT(new_var, [])
        new_cpt.entries = {tuple(): count}
        new_var.cpt = new_cpt
        new_bn.variables[v] = new_var
    new_bn.write("networks/" + network_name)


def local_search(network, max_iter, network_name):
    """
    :param network: simple Bayesian network
    :param max_iter: max interation
    :param network_name: emplacement for the improved network
    """
    cnt = 0

    best_score = network.get_alternative_score()
    print(best_score)
    network.write('best_network.bif')
    for i in range(max_iter):
        network = BayesianNetwork('best_network.bif', network.data_file)
        independent_var = [var for var in network.variables.keys() if not network.variables[var].cpt.parents]
        if not independent_var:
            break
        selected = random.choice(independent_var)
        target_var = random.choice([var for var in network.variables.keys() if var != selected])
        network.variables[selected].cpt.parents.append(network.variables[target_var])
        network.param_learning(selected, network.data_file)
        score = network.get_alternative_score()
        print(score > best_score)
        if score > best_score and cnt < 3:
            if abs(score - best_score) < 0.001:
                cnt += 1
            else:
                cnt = 0
            best_score = score
            network.write('best_network.bif')
        else:
            break
    print(f"after {best_score}")
    best = BayesianNetwork('best_network.bif', network.data_file)
    best.write(network_name)


# Example for how to read a BayesianNetwork

def missing_value_imputation(network,test_file,file):
    """
    :param network: The best bayesian network
    :param test_file: the path to the file with missing values
    :param file: datafile of the improved network
    """
    df = pd.read_csv(test_file)
    val = df.to_dict(orient='records')
       
    net = BayesianNetwork(network,file)

    for row in val:
        if any(v is None or v != v for v in row.values()):

            #values formating
            d = row
            d = {k: 1 if v == 1.0 else v for k, v in d.items()}
            d = {k: 0 if v == 0.0 else v for k, v in d.items()}
            d = {k: 2 if v == 2.0 else v for k, v in d.items()}
            d = {k: 3 if v == 3.0 else v for k, v in d.items()}

            nan_keys = [k for k, v in d.items() if isinstance(v, float)
                        and math.isnan(v)]

            not_nan_dict = {k: v for k, v in d.items() if not isinstance(
                v, float) or not math.isnan(v)}
            
            #missing value imputation here
            if len(nan_keys) == 1:
                t = net.get_distrib_givenX(nan_keys[0], not_nan_dict)
                m = max(t, key=t.get)
                row[nan_keys[0]] = float(m)
            else:
                t = net.get_distrib_givenX_double(nan_keys, not_nan_dict)
                m = max(t, key=t.get)
                v1,v2 = m
                y1,y2 = nan_keys
                row[y1] = float(v1)
                row[y2] = float(v2)

    df2 = pd.DataFrame(val)
    df2.to_csv('datasets/asia/Imputed_values.csv', index=False)

def accuracy(df_A, df_B):
    # Vérifier si les DataFrames ont la même taille
    if df_A.shape != df_B.shape:
        raise ValueError("Les DataFrames doivent avoir la même taille.")

    total_rows = df_A.shape[0]
    num_identical_rows = 0

    # Parcourir les lignes des DataFrames
    for i in range(total_rows):
        row_A = df_A.iloc[i]
        row_B = df_B.iloc[i]

        # Comparer les valeurs des deux lignes
        if row_A.equals(row_B):
            num_identical_rows += 1

    accuracy = num_identical_rows / total_rows
    return accuracy

if __name__ == '__main__':
    '''
    structure_init('asia.bif', 'datasets/asia/train.csv')
    bn = BayesianNetwork('networks/asia.bif', 'datasets/asia/train.csv')
    local_search(bn, 100, 'networks/asia.bif')
    bn = BayesianNetwork('networks/asia.bif', 'datasets/asia/train.csv')
    print(bn.get_distrib_givenX_double(['lung', 'tub'], {'smoke':0,  'asia':1, 'lung':1, 'bronc':1, 'either':0, 'xray':1, 'dysp':0}))
    print(bn.get_distrib_givenX('smoke', {'tub': 1, 'asia':1, 'lung':1, 'bronc':1, 'either':0, 'xray':1, 'dysp':0}))
    '''

    """ structure_init('asia.bif', 'datasets/asia/train.csv')
    bn = BayesianNetwork('networks/asia.bif',
                         'datasets/asia/train.csv')
    local_search(bn, 1000, 'networks/asia.bif')

    #doing the missing value imputation according to the best bayesian network
    missing_value_imputation(
        'best_network.bif', 'datasets/asia/test_missing.csv', 'datasets/asia/train.csv') """

    files = ["alarm","andes","asia","random","sachs","sprinkler","water"]
    accs = {}
    for f in files :
        df_test = pd.read_csv(f"datasets/{f}/test.csv")
        df_pred = pd.read_csv(f"datasets/{f}/Imputed_values.csv")
        df_pred =df_pred.astype('int64')
        acc = accuracy(df_test,df_pred)
        print(f"Accuracy of {acc} for the dataset {f}")
        accs[f] = acc
    keys = list(accs.keys())
    values = list(accs.values())

    # Création du graphique en barres
    plt.bar(keys, values)
    plt.xlabel('Files')
    plt.ylabel('Accuracy')
    plt.title('Bayesian network prediction accuracy')

    # Affichage du graphique
    plt.show()
    plt.savefig("plot.png")
