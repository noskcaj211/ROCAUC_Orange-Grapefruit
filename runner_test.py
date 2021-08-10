from Population import Population
from Network import Model
import torch
import csv

INPUT_SIZE = 5
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1

POPULATION_SIZE = 24
NUM_THRESHOLDS = 10
MUTATION_RATE = 0.005

THRESHOLD = 0.25

input_data_list = []
training_data_list = []

print('Getting Data...')

with open('citrus_test.csv', newline='') as csvfile:
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

model = Model(NUM_THRESHOLDS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load("Networks/population_2_150_1.pth"))

outputs = model.forward(input_data)

mse = []

tp = 0
fp = 0
tn = 0
fn = 0

results = []

for x in range(len(outputs)):
    mse.append((training_data[x].item() - outputs[x].item())**2)

    if outputs[x] >= THRESHOLD:
        #print('orange')
        result = True
    else:
        #print('grapefruit')
        result = False

    if result and training_data[x] == 1:
        tp += 1
        results.append('tp')
    elif not result and training_data[x] == 1:
        fn += 1
        results.append('fn')
    elif result and not training_data[x] == 1:
        fp += 1
        results.append('fp')
    elif not result and not training_data[x] == 1:
        tn += 1
        results.append('tn')
print()
print('True Positives:',tp)
print('False Negatives:',fn)
print('False Positives:',fp)
print('True Negative:',tn)
print()
with open('Mean_Squared_Errors.txt', 'w') as f:
    for row in mse:
        f.write(str(row) + '\n')

with open('results.txt', 'w') as f:
    for row in results:
        f.write(row + '\n')

model.run_roc(input_data, training_data)
model.calc_auc()
print('AUC', model.auc)
model.plot_roc()