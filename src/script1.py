import csv

print ('Hello World!')

#Note, port over csvfile into same /src dir to simplicity reference.
with open('un-general-debates-short.csv', 'r') as csvfile:
    freader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in freader:
        #row[0]:session
        #row[1]:year
        #row[2]:country
        #row[3]:text
        print(row[3])
    
