import json
import csv
import pandas as pd

def allFieldsPresent(jsondata):
    return len(jsondata.keys()) == 9

#LONGER/ACTUAL DATA
f=open('../datasets/amazon_review_electronic_full.json','r')
w=open('../datasets/CSV_AMAZON_REVIEW_ELECTRONIC_FULL.csv','w')

csvwriter = csv.writer(w)

rowcount=0

print('CSV Conversion Start')

for line in f:
    jsondata = json.loads(line)
    if rowcount == 0:
        header = jsondata.keys()
        print (header)
        csvwriter.writerow(header)
        
    rowcount += 1

    # Only convert if all fields are present. Some docs do not have reviewerName.
    if allFieldsPresent(jsondata):
        csvwriter.writerow(jsondata.values())
        
    if rowcount % 100000 == 0:
        print ('Processing Line:', rowcount)

    # Limiter to process smaller dataset.
    if rowcount == 1000:
        break;
    
w.close()
f.close()

print('CSV Conversion Complete')

print('Using Pandas to extract reviewText column')

df = pd.read_csv('../datasets/CSV_AMAZON_REVIEW_ELECTRONIC_FULL.csv', skipinitialspace=True)
df.index.name = 'index'
df.reviewText.to_csv('../datasets/testdata.csv', header=['reviewText'], encoding='utf-8')

print ('\n\n==END==\n\n')
