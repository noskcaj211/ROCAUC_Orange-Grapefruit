from Population import Population
import torch
import csv

INPUT_SIZE = 5
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1

POPULATION_SIZE = 20
NUM_THRESHOLDS = 24
MUTATION_RATE = 0.007



input_data_list = []
training_data_list = []

print('Getting Data...')

with open('citrus_train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if row[0] == 'orange':
            training_data_list.append(1)
        else:
            training_data_list.append(0)

        input_data_list.append([float(i) for i in row[1:]])
        
input_data = torch.Tensor(input_data_list)
training_data = torch.Tensor(training_data_list)
print('Completed Getting Data.')

population = Population(POPULATION_SIZE, NUM_THRESHOLDS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MUTATION_RATE)
#population.load(1)

epoch = 1
while epoch < 10000:
    print(epoch)
    population.run_population(input_data, training_data)

    population.sort()
    population.export_aucs(epoch)
    population.export_rocs(epoch)

    population.evolve()

    if epoch%25 == 0:
        population.save(epoch)
        
    


    epoch += 1