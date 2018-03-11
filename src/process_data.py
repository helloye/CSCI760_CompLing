import csv
import gensim

f = open('amazonReviewElectronicShortCSV.csv','r')

# csvdata = csv.reader(f)

#Index listing
# 0 - reviewerID
# 1 - asin
# 2 - reviewerName
# 3 - helpful
# 4 - reviewText (Use this as corpus?)
# 5 - overall
# 6 - summary
# 7 - unixReviewTime
# 8 - reviewTime

#for row in csvdata:
#    print "[",row[6],"] ",row[4]

f.close()
print '\n\n==END==\n\n'
