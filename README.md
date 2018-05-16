# CSCI760_CompLing
Hunter College CSCI760 Computer Linguistics Spring 2018

## Instructions/Setup

##### Python 3.6

#### Dependencies



pandas

nltk

sklearn

matplotlib

numpy

html2text

Use `pip` to download and install the above dependencies/packages.

#### Running the python scripts:

Before running the script, we must provide the dataset needed. For this project, I had used the 1.5M + Amazon reviews provided by Julian McAuley of UCSD (credited in my python notebook). Simply follow the steps below to download and load the datafile into the project:

1. Download the zipped corpus from this link: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz (if you run into an error, check my python notebook in /jupyter/notebook.ipynb for a link to the author's site.
2. Unzip the file and copy the file `review_Electronics_5.json` into the `/datasets` folder, and rename it to `amazon_review_electronic_full.json` as referenced in `/src/clean_data/py`

Once the dataset is set, you will need to run `clean_data.py` first to clean up the raw json and output a `testdata.csv` file that will be used by the main `/src/process_data.py` script. Once you have `testdata.csv` updated with the test data, simply run `/src/process_data.py` to output and visualize the results as seen in the jupyter notebook. I have left comments throughout the file to describe what each section is doing. Please feel free to reach out to me below if you have any questions

ye [at] weby3 [dot] com
