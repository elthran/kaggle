import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import sklearn
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

from utils import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ['KAGGLE_CONFIG_DIR'] = "../.kaggle/"
# !kaggle competitions download -c nlp-getting-started
# !unzip -n 'nlp-getting-started'

print("Python version:", sys.version)
print("Version info.:", sys.version_info)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("skearn version:", sklearn.__version__)

for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def load_csv(filename):
    return pd.read_csv(filename, index_col="id")


def clean_data():
    train_df = load_csv("train.csv")
    test_df = load_csv("test.csv")
    combined_df = pd.concat([train_df, test_df], sort=False)
    print(combined_df.head())
    print(combined_df.tail())

    # Remove upper case
    combined_df["text_clean"] = combined_df["text"].apply(lambda x: x.lower())

    # Remove text contractions
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: contractions.fix(x))

    # Remove any URLs from the text
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: remove_URL(x))

    # Remove any HTML tags
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: remove_html(x))

    # Remove non-ASCII from text
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: remove_non_ascii(x))

    # Remove special characters
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: remove_special_characters(x))

    # Remove punctuation
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: remove_punct(x))

    # Map common typos and slang
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: other_clean(x))

    # Fix spelling errors
    text_blob = Blobber(analyzer=NaiveBayesAnalyzer())
    combined_df["text_clean"] = combined_df["text_clean"].apply(lambda x: text_blob(x).correct())

    # Break the words into a list
    combined_df['tokenized'] = combined_df['text_clean'].apply(word_tokenize)

    # Remove stopwords (eg. About, Above, Across, After, ..
    stop = set(stopwords.words('english'))
    combined_df['stopwords_removed'] = combined_df['tokenized'].apply(lambda x: [word for word in x if word not in stop])

    # Use stemmer
    combined_df['stemmer'] = combined_df['stopwords_removed'].apply(lambda x: stemmer(x))

    combined_df.to_csv("cleaned_data.csv")


def get_clean_data():
    return load_csv("cleaned_data.csv")


clean_data()
df = get_clean_data()
print(df.head())
