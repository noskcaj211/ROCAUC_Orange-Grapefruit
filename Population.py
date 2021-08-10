from Network import Model
import random
import numpy as np
import torch

class Population:
    def __init__(self, population_size, num_thresholds, input_size, hidden_size, output_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_thresholds = num_thresholds
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.population = []
        for i in range(population_size):
            self.population.append(Model(num_thresholds, input_size, hidden_size, output_size))

    def evolve(self):
        self.kill(int(self.population_size/2))
        self.repopulate()
        return
    
    def kill(self, num):
        self.sort()
        while num > 0:
            rand = random.random()
            rand_distributed = np.sqrt(rand)
            rand_pos = int(rand_distributed * (len(self.population)-2))
            self.population.pop(rand_pos+2)
            num -= 1

    def repopulate(self):
        children = []
        while len(self.population) + len(children) < self.population_size:
            parent1 = self.population[random.randint(0,len(self.population)-1)]
            parent2 = self.population[random.randint(0,len(self.population)-1)]
            parent1_share = np.random.normal(0, 1)
            parent2_share = 1-parent1_share

            child = Model(self.num_thresholds, self.input_size, self.hidden_size, self.output_size)
            child.born(parent1, parent2, parent1_share, parent2_share, self.mutation_rate)
            children.append(child)
        self.population += children

    def sort(self):
        self.population.sort(key=lambda x: x.auc, reverse=True)

    def run_population(self, input, targets):
        for x in self.population:
            x.run_roc(input, targets)
            x.calc_auc()

    def load(self, epoch_num):
        for x in range(self.population_size):
            self.population[x].load_state_dict(torch.load("Networks/population"+"_2_"+str(epoch_num)+"_"+str(x)+".pth"))

    def save(self, epoch_num):
        for x in range(self.population_size):
            torch.save(self.population[x].state_dict(), "Networks/population"+"_2_"+str(epoch_num)+"_"+str(x)+".pth")

    def export_aucs(self, epoch_num):
        with open('AUCs/aucs'+"_2_"+str(epoch_num)+'.txt', 'w') as f:
            for x in self.population:
                f.write("%s\n" % x.auc)
    
    def export_rocs(self, epoch_num):
        with open('ROCs/rocs'+"_2_"+str(epoch_num)+'.txt', 'w') as f:
            for x in self.population:
                f.write("%s\n" % x.roc_vals)