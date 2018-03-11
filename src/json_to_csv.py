import json
import csv

#SHORT DATA
f=open('../datasets/amazon_review_electronic_short.json','r')
w=open('amazonReviewElectronicShortCSV.csv','w')

#LONGER/ACTUAL DATA
#f=open('../datasets/amazon_review_electronic_full.json','r')
#w=open('../datasets/CSV_AMAZON_REVIEW_ELECTRONIC_FULL.csv','w')

csvwriter = csv.writer(w)

rowcount=0
for line in f:
    jsondata = json.loads(line)
    if rowcount == 0:
        header = jsondata.keys()
        print header
        csvwriter.writerow(header)
        
    rowcount += 1    
    csvwriter.writerow(jsondata.values())
    if rowcount % 100000 == 0:
        print 'Processing Mark:',rowcount

w.close()
f.close()
print '\n\n==END==\n\n'
