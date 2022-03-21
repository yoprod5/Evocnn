from layers import ConvLayer, PoolLayer, FullLayer
from population import Population
from evaluate import Evaluate
from individual import Individual
import numpy
import tensorflow.examples.tutorials.mnist as input_data
import tensorflow as tf
import collections
from utils import *
import copy



class Search_Length :

    def __init__(self, train_data, train_label, validate_data, validate_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length):
        
        self.train_data = train_data # train or test data. data[0] is images and data[1] are label
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length
        ###########
        self.Length=24
        self.Space=[0,24]

    def Initialize_Length(self,N):
        self.Length=N

    def initialize_popualtion(self):
        #print("initializing population with number {}...".format(self.Length))
        self.pops = Population(0)
        self.pops.Pop_initialize_spaces(self.Length)
        print(self.pops)
        #self.pops.set_populations(self.gen_ind_spaces(self.Length))
        # all the initialized population should be saved
        save_populations(gen_no=-1, pops=self.pops)
    

    def Select_space(self, gen_no):
        assert(self.pops.get_pop_size() == 2*self.pops.get_pop_size())
        elitsam = 0.2
        e_count = int(np.floor(self.pops.get_pop_size()*elitsam/2)*2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x:x.mean, reverse=True)
        elistm_list = indi_list[0:e_count]

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        for _ in range(self.pops.get_pop_size()-e_count):
            i1 = randint(0, len(left_list))
            i2 = randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        save_populations(gen_no=gen_no, pops=self.pops)
        np.random.shuffle(self.pops.pops)

    def espace_selecion(self):  
        i=0
        notfind=1
        while (i < self.pops.get_pop_size()) and (notfind) :
            ind1=self.pops.get_individual_at(i)
            j=i+1
            while j <(self.pops.get_pop_size()):
                  ind2=self.pops.get_individual_at(j)
                  winner=self.selection(ind1,ind2)
                  if winner!=ind1 :
                     i=j
                     break
                  j+=1
            if winner==ind1 :
               notfind=0;
                    
        save_populations(gen_no=-1, pops=self.pops)
        np.random.shuffle(self.pops.pops)

        return  winner.get_length()

    def selection(self, ind1, ind2):
        mean_threshold = 0.05
        complexity_threhold = 100
        if ind1.mean > ind2.mean:
            if ind1.mean - ind2.mean > mean_threshold:
                return ind1
            else:
                if ind2.complxity < (ind1.complxity-complexity_threhold):
                    return ind2
                else:
                    return ind1
        else:
            if ind2.mean - ind1.mean > mean_threshold:
                return ind2
            else:
                if ind1.complxity < (ind2.complxity-complexity_threhold):
                    return ind1
                else:
                    return ind2

    def Search_length(self):
        
        '''
        Have to return here the optimal Sapce L=[min, max] we will find the best individu
        '''
        Len =[]
        #N= self.Length  
        self.initialize_popualtion()
        E= Evaluate(self.pops,self.train_data ,self.train_label,self.validate_data,self.validate_label,
self.number_of_channel,self.epochs,self.batch_size,self.train_data_length,self.validate_data_length) 
        E.parse_population_spaces(-1)
        Len=self.espace_selecion()
        print("L'espace optimal trouver est {} ".format(Len))
        return Len
        



            


    

