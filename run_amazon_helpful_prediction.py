import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from models.ClassifierRunner import ClassifierRunner
import datetime
import pandas as pd
import logging
import sys

# configure logger so we can see output from the classes
logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s %(levelname)s - %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

# set global variables

# I'm finding that running these models on my laptop takes forever and they are not finishing so I'm going to start
# with a really small file just to validate my code
#
# datafile was generated from amazon_review_preprocessing.ipynb - this file has 1k reviews randomly chosen
# from original file
KEEP_COLUMNS = ["helpful_votes", "total_votes", "helpful_votes", "review_headline", "review_body"]
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
OUTCOME_COLUMN = "star_rating"


# Configuration
DATA_FILE = "dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-smallout.csv"
NEIGHBORS = [5] # default
# NEIGHBORS = [1, 3, 5, 7, 9, 11]

# Radius for RadiusNeighbor
# RADII = [5.0] # this is the lowest number I tried that was able to find a neighbor for review_headline
RADII = [30.0] # this is the lowest number I tried that was able to find a neighbor for review_body
# RADII = [5.0, 7.0, 9.0, 11.0, 13.0]

# logistic regression settings
C= [1.0] # default
# C = [0.2, 0.4, 0.6, 0.8, 1.0]

N_JOBS=6
LR_ITER=500
FEATURE_COLUMN = "review_body"
CLASS_COLUMN = "helpful_votes"

# model flags
ENABLE_BOW_KNN = True
ENABLE_BOW_RN = True
ENABLE_BOW_LR = True

ENABLE_TFIDF_KNN = True
ENABLE_TFIDF_RN = True
ENABLE_TFIDF_LR = True




WRITE_TO_CSV = True


# In[3]:


# read in DF
df = pd.read_csv(DATA_FILE)[KEEP_COLUMNS]
print(df.info())
df.head()


# ### <font color="red">Should I include add these along with the word count as part of the feature set?</font>

# In[4]:


# let's get some data on our text

def wc(x:str):
    return len(str(x).split())

df["rh_wc"] = df.review_headline.apply(wc)
df["rb_wc"] = df.review_body.apply(wc)
df.describe()


# In[5]:


# Set up different dataframes for training

# outcome
helpful_df = df[df["helpful_votes"] > 0]
sys.exit(0)
X = df[df["helpful_votes"] > 0][FEATURE_COLUMN]
Y = df.iloc[X.index][CLASS_COLUMN]




# TODO: try different parameters for CountVectorizers?
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(X.array)
vocab = cv.get_feature_names()
# print(f"vocab: {vocab}")
bag_pd = pd.DataFrame(cv_matrix.toarray(), columns=vocab)

# split results into training and test set
bag_X_train, bag_X_test, bag_Y_train, bag_Y_test = train_test_split(bag_pd, Y, random_state=1)

print(f"training set size {len(bag_X_train)}")
print(f"test set size {len(bag_X_test)}")


# In[7]:


# explore the data
print(len(vocab))
bag_pd.head()





# # Set up BoW models

# In[ ]:

start_time = datetime.now()

cr = ClassifierRunner(write_to_csv=WRITE_TO_CSV)

if ENABLE_BOW_KNN:
    for neighbor in NEIGHBORS:
        neigh = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=N_JOBS)
        cr.addModel(neigh,
                    bag_X_train,
                    bag_Y_train,
                    bag_X_test,
                    bag_Y_test,
                    name="KNN",
                    dataset="BoW",
                    parameters={"n_jobs": N_JOBS,
                               "n_neighbors": neighbor})

if ENABLE_BOW_RN:
    for radius in RADII:
        rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=N_JOBS)
        cr.addModel(neigh,
                    bag_X_train,
                    bag_Y_train,
                    bag_X_test,
                    bag_Y_test,
                    name="RN",
                    dataset="BoW",
                    parameters={"n_jobs": N_JOBS,
                               "radius": radius})

if ENABLE_BOW_LR:
    for c in C:
        lr = LogisticRegression(random_state=0, solver='lbfgs',
                                multi_class='auto',
                                max_iter=LR_ITER, n_jobs=N_JOBS, C=c)
        cr.addModel(lr,
                    bag_X_train,
                    bag_Y_train,
                    bag_X_test,
                    bag_Y_test,
                    name="LR",
                    dataset="BoW",
                    parameters={"n_jobs": N_JOBS,
                               "c": c,
                               "max_iter": LR_ITER})


# In[ ]:


report_df = cr.run_models()
report_df.head()


# # TFIDF - default settings

# ### Feature Generation

# In[ ]:


# TODO: play with min_df and max_df
# TODO: play with variations of ngram
tv = TfidfVectorizer(min_df=0., max_df=1., ngram_range=(1,3), use_idf=True)
tv_matrix = tv.fit_transform(X.array)
vocab = tv.get_feature_names()
tv_pd = pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=vocab)

# split results into training and test set
tv_X_train, tv_X_test, tv_Y_train, tv_Y_test = train_test_split(tv_pd, Y, random_state=1)

print(f"training set size {len(tv_X_train)}")
print(f"test set size {len(tv_X_test)}")


# ### Set Up Models

# In[ ]:


if ENABLE_TFIDF_KNN:
    for neighbor in NEIGHBORS:
        neigh = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=N_JOBS)
        cr.addModel(neigh,
                    tv_X_train,
                    tv_Y_train,
                    tv_X_test,
                    tv_Y_test,
                    name="KNN",
                    dataset="TFIDF",
                    parameters={"n_jobs": N_JOBS,
                               "n_neighbors": neighbor})

if ENABLE_TFIDF_RN:
    for radius in RADII:
        rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=N_JOBS)
        cr.addModel(neigh,
                    tv_X_train,
                    tv_Y_train,
                    tv_X_test,
                    tv_Y_test,
                    name="RN",
                    dataset="TFIDF",
                    parameters={"n_jobs": N_JOBS,
                               "radius": radius})

if ENABLE_TFIDF_LR:
    for c in C:
        lr = LogisticRegression(random_state=0, solver='lbfgs',
                                multi_class='auto',
                                max_iter=LR_ITER, n_jobs=N_JOBS, C=c)
        cr.addModel(lr,
                    tv_X_train,
                    tv_Y_train,
                    tv_X_test,
                    tv_Y_test,
                    name="LR",
                    dataset="TFIDF",
                    parameters={"n_jobs": N_JOBS,
                               "c": c,
                               "max_iter": LR_ITER})

report_df = cr.runNewModels()
log.info("Finished running all models")
report_df.head()


end_time = datetime.now()


total_time = end_time - start_time
log.info(f'Total time: {total_time.strftime(TIME_FORMAT)}')

