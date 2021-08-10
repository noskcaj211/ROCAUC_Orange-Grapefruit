from Population import Population
from Network import Model
import torch
import matplotlib.pyplot as plt

def plot_roc(data, epoch):
    
    x = [x[0] for x in data]
    y = [y[1] for y in data]
    plt.plot(x,y, label=('Epoch:'+str(epoch)))
    plt.legend()
    plt.show()
    plt.savefig('BestMembers/curve_'+str(epoch))
    plt.close()


def plot_big_roc(data):
    i = 1
    for member in data:
        x = [x[0] for x in member]
        y = [y[1] for y in member]
        plt.plot(x,y, label=('Epoch:'+str(i)))
        i+=1
    plt.show()
    plt.savefig('Best Members Big ROC')
    plt.close()

best_members = []
worst_members = []

for x in range(1, 150):
    with open('ROCs/rocs_2_'+str(x)+'.txt', 'r')as f:
        x = f.readline()[2:-3]
        x = x.split('], [')
        temp_list = []
        for e in x:
            e =  e.split(', ')
            temp_list.append([float(e[0]),float(e[1])])
        best_members.append(temp_list)
        for x in f:
            pass
        x = x[2:-3]
        x = x.split('], [')
        temp_list = []
        for e in x:
            e = e.split(', ')
            temp_list.append([float(e[0]),float(e[1])])
        worst_members.append(temp_list)

plot_big_roc(best_members)

i = 1
for e in best_members:
    plot_roc(e, i)
    i+=1