import re

# A class for representing the CPT of a variable
class CPT:

    def __init__(self, head, parents):
        self.head = head # The variable this CPT belongs to (object)
        self.parents = parents # Parent variables (objects), in order
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
                f"  table {comma.join ( map(str,self.entries[tuple()].values () ))};" + "\n" \
                f"}}" + "\n"
        else:
            return f"probability ( {self.head.name} | {comma.join ( [p.name for p in self.parents ] )} ) {{" + "\n" + \
                "\n".join ( [  \
                  f"  ({comma.join(names)}) {comma.join(map(str,values.values ()))};" \
                    for names,values in self.entries.items () \
                ] ) + "\n}\n" 


# A class for representing a variable
class Variable:
    
    def __init__(self, name, values):
        self.name = name # Name of the variable
        self.values = values 
          # The domain of the variable: names of the values
        self.cpt = None # No CPT initially

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
                assert(variable_def[1] == 'discrete')
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

            assert(variable.name == split[2])

            cpt = CPT(variable, parents)
            i += 1
            
            nb_lines = 1
            for p in parents:
                nb_lines *= len(p.values)
            for lid in range(nb_lines):
                cpt_line = lines[i].split(' ')
                #parent_values = [parents[j].values[re.sub('\(|\)|,', '', cpt_line[j])] for j in range(len(parents))]
                parent_values = tuple([ re.sub('\(|\)|,', '', cpt_line[j]) for j in range(len(parents)) ])
                probabilities = re.findall("\d\.\d+(?:e-\d\d)?", lines[i])
                cpt.entries[parent_values] = { v:float(p) for v,p in zip(variable.values,probabilities) }
                i += 1
            variable.cpt = cpt
            i += 1

    # Method for writing a Bayesian Network to an output file
    def write(self,filename):
        with open(filename,"w") as file:
            for var in self.variables.values ():
                file.write(str(var))
            for var in self.variables.values ():
                file.write(str(var.cpt))

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and x an assignment to its parents in the network, specified
    # in the correct order of parents.
    def P_Yisy_given_parents_x(self,Y,y,x=tuple()):
        return self.variables[Y].cpt.entries[x][y]

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and pa a dictionnary of assignments to the parents,
    # with for every parent variable its associated assignment.
    def P_Yisy_given_parents(self,Y,y,pa={}):
        x = tuple([ pa[parent.name] for parent in self.variables[Y].cpt.parents ])
        return self.P_Yisy_given_parents_x(Y,y,x)


# Example for how to read a BayesianNetwork
bn = BayesianNetwork ( "alarm.bif" )

# Example for how to write a BayesianNetwork
bn.write("alarm2.bif")

# Examples for how to get an entry from the CPT

# return P(HISTORY=TRUE|LVFAILURE=TRUE)
print(bn.P_Yisy_given_parents_x("HISTORY","TRUE",("TRUE",)))
# or
print(bn.P_Yisy_given_parents("HISTORY","TRUE",{"LVFAILURE":"TRUE"}))

# return P(HRBP=NORMAL|ERRLOWOUTPUT=TRUE,HR=LOW)
print(bn.P_Yisy_given_parents_x("HRBP","NORMAL",("TRUE","LOW")))
# or
print(bn.P_Yisy_given_parents("HRBP","NORMAL",{"ERRLOWOUTPUT":"TRUE","HR":"LOW"}))

# return P(HYPOVOLEMIA=TRUE)
print(bn.P_Yisy_given_parents_x("HYPOVOLEMIA","TRUE"))
# or
print(bn.P_Yisy_given_parents("HYPOVOLEMIA","TRUE"))
