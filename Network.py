import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ConfusionMatrix():
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

class Model(nn.Module):
    def __init__(self, num_thresholds, input_size, hidden_size, output_size):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear_hiddens = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

        self.num_thresholds = num_thresholds
        self.auc = 0
        if num_thresholds < 2:
            raise Exception('Invalid number of thresholds')
        self.num_thresholds = num_thresholds
        self.thresholds = []
        
        difference = 1 / (num_thresholds-1)
        i = 0
        while i < 1-difference/2:
            self.thresholds.append(i)
            i += difference
        self.thresholds.append(1)
        self.roc_confusions = []
        for i in range(self.num_thresholds):
            self.roc_confusions.append(ConfusionMatrix())
        self.roc_vals = [[0,0]]*self.num_thresholds


    def forward(self, inp):
        output1 = self.linear_hiddens(inp)
        output1 = self.linear_output(output1)
        return self.sigmoid(output1)

    def run_roc(self, input, targets):
        results = self.forward(input)
        for x in range(len(results)):
            for i in range(len(self.thresholds)):
                result = results[x] >= self.thresholds[i]
                if result and targets[x] == 1:
                    self.roc_confusions[i].tp += 1
                elif not result and targets[x] == 1:
                    self.roc_confusions[i].fn += 1
                elif result and not targets[x] == 1:
                    self.roc_confusions[i].fp += 1
                elif not result and not targets[x] == 1:
                    self.roc_confusions[i].tn += 1
        
        self.roc_vals.clear()

        for matrix in self.roc_confusions:
            fpr = matrix.fp/(matrix.fp+matrix.tn)
            tpr = matrix.tp/(matrix.tp+matrix.fn)
            self.roc_vals.append([fpr, tpr])

    #performs trapezoidal approximation on the roc curve values
    def calc_auc(self):
        auc = 0
        self.roc_vals.sort(key = lambda x: x[1])
        self.roc_vals.sort(key = lambda x: x[0])
        for x in range(len(self.roc_vals)-1):
            auc += (self.roc_vals[x][1] + self.roc_vals[x+1][1])/2*(self.roc_vals[x+1][0] - self.roc_vals[x][0])

        self.auc = auc


    def born(self, parent1, parent2, parent1_share, parent2_share, mutation_rate):
        self.linear_hiddens.weight.data = (parent1.linear_hiddens.weight.data * parent1_share) + (parent2.linear_hiddens.weight.data * parent2_share)
        self.linear_hiddens.bias.data = (parent1.linear_hiddens.bias.data * parent1_share) + (parent2.linear_hiddens.bias.data * parent2_share)
        self.linear_output.weight.data = (parent1.linear_output.weight.data * parent1_share) + (parent2.linear_output.weight.data * parent2_share)
        self.linear_output.bias.data = (parent1.linear_output.bias.data * parent1_share) + (parent2.linear_output.bias.data * parent2_share)
        self.mutate(mutation_rate)
        
    def mutate(self, mutation_rate):
        self.linear_hiddens.weight.data += mutation_rate * (torch.rand_like(self.linear_hiddens.weight.data) * 2 - 1)
        self.linear_hiddens.bias.data += mutation_rate * (torch.rand_like(self.linear_hiddens.bias.data) * 2 - 1)
        self.linear_output.weight.data += mutation_rate * (torch.rand_like(self.linear_output.weight.data) * 2 - 1)
        self.linear_output.bias.data += mutation_rate * (torch.rand_like(self.linear_output.bias.data) * 2 - 1)

    def plot_roc(self):
        self.roc_vals.sort(key = lambda x: x[1])
        self.roc_vals.sort(key = lambda x: x[0])
        x = [x[0] for x in self.roc_vals]
        y = [y[1] for y in self.roc_vals]
        plt.plot(x,y, label=('AUC:'+str(self.auc)))
        plt.legend()
        plt.show()
        plt.savefig('ROC Curve')