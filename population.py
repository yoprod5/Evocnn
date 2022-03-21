from individual import Individual
from utils import *

class Population:

    def __init__(self, num_pops):
        self.num_pops = num_pops
        
        
    def Pop_initialize(self,length):
        print("initializing population with number {}...".format(length))
        self.pops = []
        for _ in range(self.num_pops):
            indi = Individual()
            indi.initialize(length)
            self.pops.append(indi)
    
    def Pop_initialize_spaces(self,length):

        print("initializing population with number {}...".format(length))

        self.pops = []
        self.num_pops=int(length/4)
        #indi = Individual()
        #indi.initialize_spaces(int(2))
        #self.pops.append(indi)
        indi = Individual()
        indi.initialize_spaces(0)
        self.pops.append(indi)
        for i in range(self.num_pops+1):
            if i != 0 :   
               Ncp=2*(2*i-1)
               indi = Individual()
               indi.initialize_spaces(int(Ncp))
               self.pops.append(indi)
        indi = Individual()
        indi.initialize_spaces(4)
        self.pops.append(indi)
       
      

    def get_individual_at(self, i):
        return self.pops[i]

    def get_pop_size(self):
        return len(self.pops)

    def set_populations(self, new_pops):
        self.pops = new_pops





    def __str__(self):
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)
