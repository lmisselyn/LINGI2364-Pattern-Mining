import re
import itertools
import pandas as pd


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
    def __init__(self, input_file):

        with open(input_file) as f:
            lines = f.readlines()

        self.variables = {}
        # The dictionary of variables, allowing quick lookup from a variable
        # name.
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace('/', '-')

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
        p = 1
        for var in transaction.keys():
            var_parents = {}
            for pa in self.variables[var].cpt.parents:
                var_parents[pa.name] = transaction[pa.name]
            p_Xisx = self.P_Yisy_given_parents(var, transaction[var], var_parents)
            p = p * p_Xisx
        return p

    def get_distrib_givenX(self, Y, x={}):
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

    def get_distrib_givenX_V2(self, Y, x={}):
        res_distrib = {}
        combinaisons = self.get_combinations(Y)
        # denominator
        p_denominator = 0
        for dic in combinaisons:
            transac = x.copy()
            transac[Y[0]] = dic[Y[0]]
            transac[Y[1]] = dic[Y[1]]
            p_denominator += self.P_transaction(transac)
        # numerator
        for dic in combinaisons:
            transac = x.copy()
            transac[Y[0]] = dic[Y[0]]
            transac[Y[1]] = dic[Y[1]]
            p_numerator = self.P_transaction(transac)
            res_distrib[str([dic[Y[0]], dic[Y[1]]])] = p_numerator / p_denominator
        return res_distrib

    def get_combinations(self, Y):
        combi = []
        for y1 in self.variables[Y[0]].values:
            for y2 in self.variables[Y[1]].values:
                combi.append({Y[0]: y1, Y[1]: y2})
        return combi

    def get_combinations2(self, variables):
        arr = []
        if len(variables) == 1:
            for i in self.variables[variables[0].name].values:
                arr.append([i])
            return arr

        for v in variables:
            arr.append(self.variables[v.name].values)
        combinations = list(itertools.product(*arr))
        return combinations

    def param_learning(self, variable, filename):
        new_entry = {}
        df = pd.read_csv(filename)
        parents = self.variables[variable].cpt.parents
        combinations = self.get_combinations2(parents)

        for assignment in combinations:
            int_values = []
            for i in range(len(parents)):
                p_values = parents[i].values
                print(p_values)
                for j in range(len(p_values)):
                    if p_values[j] == assignment[i]:
                        int_values.append(j)
            print(int_values)
            #for value in variable.values:
            data = df.copy()
            for i in range(len(parents)):
                data = data.loc[df[parents[i].name] == int_values[i]]
            print(data[parents[0].name])





def structure_init(filename, network_name):
    values = {2: "{ TRUE, FALSE };", 3: "{ LOW, NORMAL, HIGH };", 4: "{ ZERO, LOW, NORMAL, HIGH };"}
    var_cardinality = {}
    f = open("networks/" + network_name, "w")
    df = pd.read_csv(filename)
    variables = df.columns
    for v in variables:
        cardinality = len(df[v].unique())
        var_cardinality[v]=cardinality
        f.write("variable " + v + " {\n  type discrete [ " + str(cardinality) + " ] " + values[cardinality] + "\n}\n")
    for v in variables:
        c = var_cardinality[v]
        count = {}
        for i in range(c):
            count[i] = 0
        for i in df[v].values:
            count[i] = count[i]+1
        for k in count.keys():
            count[k] = count[k]/len(df[v].values)
        probs = ''
        for i in range(c):
            if i != c-1:
                probs += str(count[i]) + ', '
            else:
                probs += str(count[i]) + ';'
        f.write('probability ( '+v+' ) {\n  table '+probs+'\n}\n')




# Example for how to read a BayesianNetwork
# bn = BayesianNetwork("test.bif")

# print(bn.get_distrib_givenX_V2(['FLU', 'FEVER'], {'FATIGUE': 'TRUE'}))

if __name__ == '__main__':
    #structure_init('datasets/alarm/train.csv', 'alarm_test.bif')
    bn = BayesianNetwork('alarm.bif')
    bn.param_learning('VENTMACH', 'datasets/alarm/train.csv')

