import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import sklearn
import textblob

from helpers import *

os.environ['KAGGLE_CONFIG_DIR'] = "../.kaggle/"
# !kaggle competitions download -c nlp-getting-started
# !unzip -n 'nlp-getting-started'

print("Python version:", sys.version)
print("Version info.:", sys.version_info)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("skearn version:", sklearn.__version__)
print("re version:", re.__version__)
print("nltk version:", nltk.__version__)

for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# read the csv file
train_df = pd.read_csv("train.csv")



# Remove upper case
train_df["text_clean"] = train_df["text"].apply(lambda x: x.lower())

# Remove text contractions
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: contractions.fix(x))

# Remove any URLs from the text
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_URL(x))

# Remove any HTML tags
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_html(x))

#Remove non-ASCII from text
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_non_ascii(x))

# Remove special characters
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_special_characters(x))

# Remove punctuation
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_punct(x))

# Map common typos and slang
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: other_clean(x))

# Fix spelling errors
train_df["text_clean"] = train_df["text_clean"].apply(lambda x: textblob.TextBlob(x).correct())

# Break the words into a list
train_df['tokenized'] = train_df['text_clean'].apply(nltk.tokenize.word_tokenize)

# Remove stopwords (eg. About, Above, Across, After, ..
train_df['stopwords_removed'] = train_df['tokenized'].\
    apply(lambda x: [word for word in x if word not in nltk.corpus.stop])

# Use stemmer
train_df['stemmer'] = train_df['stopwords_removed'].apply(lambda x: stemmer(x))




