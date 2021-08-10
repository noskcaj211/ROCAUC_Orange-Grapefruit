import csv


tot = 0
count = 0
with open('Mean_Squared_Errors.txt', newline='') as f:
    for row in f:
        tot += float(row)
        count+=1

print(tot/count)