"""
Reads in one of the feature files and creates and output csv with the distribution
of star_ratings so that we can graph it in our notebook
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# TODO: Update here
print("Reading feature file")
# in_df = pd.read_csv("dataset/feature_files/review_body-bow-df_default-ngram11-992-3322.csv")
in_df = pd.read_csv("dataset/feature_files/review_body-bow-df_default-ngram11-111909-10000.csv")
print("Finished feature file")
out_df = in_df.groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
print("Finished group by")
out_df.to_csv("dataset/feature_files/111909-hist.csv", index=False)
# out_df.to_csv("dataset/feature_files/992-hist.csv", index=False)

grouped_df = in_df.groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
count_1 = int(round(grouped_df.loc[0, "count"] * 1.66))
count_2 = int(round(grouped_df.loc[1, "count"] * 1.66))
count_3 = int(round(grouped_df.loc[2, "count"] * 1.66))
count_4 = int(round(grouped_df.loc[3, "count"] * 1.66))

y = in_df["star_rating"]
X = in_df.drop(["star_rating", "helpful_votes", "total_votes"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


sm = SMOTE(random_state=2, sampling_strategy={1: count_1, 2: count_2, 3: count_3, 4: count_4})
print("starting SMOTE")
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
print("finished SMOTE")
# print(type(X_train_res))
# print(X_train_res.shape)
# print(y_train_res.shape)
X_train_pd = pd.DataFrame(X_train_res)
Y_train_pd = pd.DataFrame(y_train_res, columns=["star_rating"])
# print(X_train_pd.info())
# print(Y_train_pd.info())
df3 = X_train_pd.join(Y_train_pd).groupby("star_rating").size().reset_index().rename(mapper={0: "count"}, axis=1)
print("Finished group by")
df3.to_csv("dataset/feature_files/111909smote-hist.csv", index=False)
