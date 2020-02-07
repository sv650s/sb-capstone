"""
Use this program to generate a vocab list and word occurence for a corpus


This will use CountTokenizer to parse out all word tokens, then we will sum up word
occurrences in our corpus

The output file will be sorted with the most frequently occuring tokens at the beginning
of the file and the least at the end of the file

Input:
* CSV file - you will have to specify which column to generate the vocab list from

Output:
* CVS with a list of vocabulary and how often they occur in the corpus
"""
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import logging


logging.basicConfig(level=logging.ERROR)



def _get_output_filename(infile_length:int , outdir:str = "../reports", output_description:str = None):
    if output_description is not None:
        return f'{outdir}/corpus-word-count-{infile_length}-{output_description}.csv'
    else:
        return f'{outdir}/corpus-word-count-{infile_length}.csv'





def main(infile: str,
         output_dir: str,
         column: str,
         output_description :str = None):

    print(f'Reading input file: {infile}')
    # read in data file
    df = pd.read_csv(infile)
    print(f'Corpus length: {len(df)}')
    # parse out column with corpus
    corpus = df[column]

    # use default setting since we want all words in the entire corpus
    print(f'Tokenizing corpus...')
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(corpus.array)

    vocab = cv.get_feature_names()
    print(f'Total vocab_length: {len(vocab)}')

    # this has all words as columns and bow count as features
    df = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

    # convert this into a DF where we have words and their occurrences (sum of bow count)
    wc_df = pd.DataFrame(df.sum()).reset_index(). \
        rename({0: "occurrence", "index": "word"}, axis=1). \
        sort_values("occurrence", ascending=False)

    # write out putput file
    outfile = _get_output_filename(len(df),
                                   output_dir,
                                   output_description)
    print(f'Writing to output file: {outfile}')
    wc_df.to_csv(outfile, index=False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Takes an input file with one of it's columns "
                                    "as the corpus and generates an output file with all vocabulary and "
                                     "their occurrence count in the corpus.")
    parser.add_argument("-c",
                        "--column",
                        help="input file (preprocessed file)",
                        default="review_body")
    parser.add_argument("-d", "--output_description",
                        help="output description - this will be part of the output file name")
    parser.add_argument("-o", "--output_dir",
                        help="outfile directory. default: ../reports",
                        default="../reports")
    parser.add_argument("infile", help="input file (preprocessed file)")

    args = parser.parse_args()

    main(args.infile,
          args.output_dir,
          args.column,
          args.output_description)










