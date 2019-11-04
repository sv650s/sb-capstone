"""
Reads in one of the feature files and creates and output csv with the distribution
of star_ratings so that we can graph it in our notebook
"""
import sys
sys.path.append('../')


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse
import logging

LOG_FORMAT='%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'

parser = argparse.ArgumentParser(description='Add a class column ')
parser.add_argument("infile", help="input datafile")
parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
# get command line arguments
args = parser.parse_args()

# process argument
if args.loglevel is not None:
    loglevel = getattr(logging, args.loglevel.upper(), None)
logging.basicConfig(format=LOG_FORMAT, level=loglevel)
log = logging.getLogger(__name__)


# TODO: Update here
log.info(f"Reading feature file {args.infile}")
# in_df = pd.read_csv("dataset/feature_files/review_body-bow-df_default-ngram11-992-3322.csv")
in_df = pd.read_csv(args.infile)
row, col = in_df.shape
log.info("Finished reading feature file")
out_df = in_df.groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
log.info("Finished group by")
out_df.to_csv(f"dataset/feature_files/{row}-hist.csv", index=False)
# out_df.to_csv("dataset/feature_files/992-hist.csv", index=False)

grouped_df = in_df.groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
count_1 = int(round(grouped_df.loc[0, "count"] * 1.66))
count_2 = int(round(grouped_df.loc[1, "count"] * 2))
count_3 = int(round(grouped_df.loc[2, "count"] * 1.9))
count_4 = int(round(grouped_df.loc[3, "count"] * 1.66))

y = in_df["star_rating"]
X = in_df.drop(["star_rating", "helpful_votes", "total_votes"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


sm = SMOTE(random_state=2, sampling_strategy={1: count_1, 2: count_2, 3: count_3, 4: count_4})
log.info("starting SMOTE")
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
log.info("finished SMOTE")
# print(type(X_train_res))
# print(X_train_res.shape)
# print(y_train_res.shape)
X_train_pd = pd.DataFrame(X_train_res)
Y_train_pd = pd.DataFrame(y_train_res, columns=["star_rating"])
# print(X_train_pd.info())
# print(Y_train_pd.info())
df3 = X_train_pd.join(Y_train_pd).groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
log.info("Finished group by")
df3.to_csv(f"dataset/feature_files/{row}smote-hist.csv", index=False)
