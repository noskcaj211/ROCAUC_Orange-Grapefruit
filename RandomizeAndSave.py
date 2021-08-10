import csv
import random
#name,diameter,weight,red,green,blue
rows = []
with open('citrus.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
       rows.append(row)

random.shuffle(rows)

with open('citrus_train.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for row in rows:
        spamwriter.writerow(row)