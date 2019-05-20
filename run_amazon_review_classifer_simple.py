import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from models.ClassifierRunner import ClassifierRunner
from datetime import datetime
import pandas as pd
import logging

# configure logger so we can see output from the classes
logging.basicConfig(format='%(asctime)s %(name)s.%(funcName)s %(levelname)s - %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

# set global variables

# I'm finding that running these models on my laptop takes forever and they are not finishing so I'm going to start
# with a really small file just to validate my code
#
# datafile was generated from amazon_review_preprocessing.ipynb - this file has 1k reviews randomly chosen
# from original file
KEEP_COLUMNS = ["product_title", "helpful_votes", "review_headline", "review_body", "star_rating"]
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
FILE_DATE_FORMAT = '%Y-%m-%d-%H'


# Configuration
DATA_FILES = ["dataset/amazon_reviews/amazon_reviews_us_Wireless_v1_00-tinyout.csv"]
NEIGHBORS = [5] # default
# NEIGHBORS = [1, 3, 5, 7, 9, 11]

# Radius for RadiusNeighbor
# RADII = [5.0] # this is the lowest number I tried that was able to find a neighbor for review_headline
RADII = [30.0] # this is the lowest number I tried that was able to find a neighbor for review_body
# RADII = [5.0, 7.0, 9.0, 11.0, 13.0]

# logistic regression settings
C= [1.0] # default
# C = [0.2, 0.4, 0.6, 0.8, 1.0]

N_JOBS=-1
LR_ITER=500
FEATURE_COLUMN = "review_headline"
CLASS_COLUMN = "star_rating"

# model flags
ENABLE_BOW_KNN = False
ENABLE_BOW_RN = False
ENABLE_BOW_LR = False

ENABLE_TFIDF_KNN = False
ENABLE_TFIDF_RN = False
ENABLE_TFIDF_LR = False

# settings min_df = 0.5 and max_df = 0.95
ENABLE_BOW90_KNN = True
ENABLE_BOW90_RN = True
ENABLE_BOW90_LR = True

# settings min_df = 0.5 and max_df = 0.95
ENABLE_TFIDF90_KNN = True
ENABLE_TFIDF90_RN = True
ENABLE_TFIDF90_LR = True


WRITE_TO_CSV = True
OUTFILE=f'{datetime.now().strftime(FILE_DATE_FORMAT)}-amazon_review-{FEATURE_COLUMN}-report.csv'



cr = ClassifierRunner(write_to_csv=WRITE_TO_CSV, outfile=OUTFILE)
# In[3]:
for data_file in DATA_FILES:


    # read in DF
    df = pd.read_csv(data_file)[KEEP_COLUMNS]
    print(df.info())
    df.head()


    # ### <font color="red">Should I include add these along with the word count as part of the feature set?</font>

    # In[4]:


    # let's get some data on our text

    def wc(x:str):
        return len(str(x).split())

    df["pt_wc"] = df.product_title.apply(wc)
    df["rh_wc"] = df.review_headline.apply(wc)
    df["rb_wc"] = df.review_body.apply(wc)
    df.describe()


    # In[5]:


    # Set up different dataframes for training

    # outcome
    Y = df[CLASS_COLUMN]
    X = df[FEATURE_COLUMN]


    # # Bag of Words - Generate Feature Vectors

    # In[6]:






    # # Set up BOW models

    # In[ ]:

    start_time = datetime.now()

    # TODO: try different parameters for CountVectorizers?
    cv90 = CountVectorizer(min_df=0.05, max_df=0.95)
    cv90_matrix = cv90.fit_transform(X.array)
    vocab90 = cv90.get_feature_names()
    vocab90_length = len(vocab90)
    log.info(f'vocab90_length: {vocab90_length}')
    # print(f"vocab: {vocab}")
    bag90_pd = pd.DataFrame(cv90_matrix.toarray(), columns=vocab90)

    # split results into training and test set
    bag90_X_train, bag90_X_test, bag90_Y_train, bag90_Y_test = train_test_split(bag90_pd, Y, random_state=1)

    if ENABLE_BOW90_KNN:
        for neighbor in NEIGHBORS:
            neigh = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=N_JOBS)
            cr.addModel(neigh,
                        bag90_X_train,
                        bag90_Y_train,
                        bag90_X_test,
                        bag90_Y_test,
                        name="KNN",
                        dataset="BOW90",
                        parameters={"n_jobs": N_JOBS,
                                    "n_neighbors": neighbor,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    if ENABLE_BOW90_RN:
        for radius in RADII:
            rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=N_JOBS)
            cr.addModel(neigh,
                        bag90_X_train,
                        bag90_Y_train,
                        bag90_X_test,
                        bag90_Y_test,
                        name="RN",
                        dataset="BOW90",
                        parameters={"n_jobs": N_JOBS,
                                    "radius": radius,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    if ENABLE_BOW90_LR:
        for c in C:
            lr = LogisticRegression(random_state=0, solver='lbfgs',
                                    multi_class='auto',
                                    max_iter=LR_ITER, n_jobs=N_JOBS, C=c)
            cr.addModel(lr,
                        bag90_X_train,
                        bag90_Y_train,
                        bag90_X_test,
                        bag90_Y_test,
                        name="LR",
                        dataset="BOW90",
                        parameters={"n_jobs": N_JOBS,
                                    "c": c,
                                    "max_iter": LR_ITER,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    report_df = cr.runNewModels()



    log.info("Creating TFIDF90 vector")
    tv90 = TfidfVectorizer(min_df=0.05, max_df=0.95, ngram_range=(1,3), use_idf=True)
    tv90_matrix = tv90.fit_transform(X.array)
    vocab90 = tv90.get_feature_names()
    vocab90_length = len(vocab90)
    tv90_pd = pd.DataFrame(np.round(tv90_matrix.toarray(), 2), columns=vocab90)
    log.info("Finished creating TFIDF90 vector")

    # split results into training and test set
    tv90_X_train, tv90_X_test, tv90_Y_train, tv90_Y_test = train_test_split(tv90_pd, Y, random_state=1)

    print(f"training set size {len(tv90_X_train)}")
    print(f"test set size {len(tv90_X_test)}")

    if ENABLE_TFIDF90_KNN:
        for neighbor in NEIGHBORS:
            neigh = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=N_JOBS)
            cr.addModel(neigh,
                        tv90_X_train,
                        tv90_Y_train,
                        tv90_X_test,
                        tv90_Y_test,
                        name="KNN",
                        dataset="TFIDF90",
                        parameters={"n_jobs": N_JOBS,
                                    "n_neighbors": neighbor,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    if ENABLE_TFIDF90_RN:
        for radius in RADII:
            rnc = RadiusNeighborsClassifier(radius=radius, n_jobs=N_JOBS)
            cr.addModel(neigh,
                        tv90_X_train,
                        tv90_Y_train,
                        tv90_X_test,
                        tv90_Y_test,
                        name="RN",
                        dataset="TFIDF90",
                        parameters={"n_jobs": N_JOBS,
                                    "radius": radius,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    if ENABLE_TFIDF90_LR:
        for c in C:
            lr = LogisticRegression(random_state=0, solver='lbfgs',
                                    multi_class='auto',
                                    max_iter=LR_ITER, n_jobs=N_JOBS, C=c)
            cr.addModel(lr,
                        tv90_X_train,
                        tv90_Y_train,
                        tv90_X_test,
                        tv90_Y_test,
                        name="LR",
                        dataset="TFIDF90",
                        parameters={"n_jobs": N_JOBS,
                                    "c": c,
                                    "max_iter": LR_ITER,
                                    "min_df": 0.05,
                                    "max_df": 0.95})

    report_df = cr.runNewModels()




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


    if ENABLE_BOW_KNN:
        for neighbor in NEIGHBORS:
            neigh = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=N_JOBS)
            cr.addModel(neigh,
                        bag_X_train,
                        bag_Y_train,
                        bag_X_test,
                        bag_Y_test,
                        name="KNN",
                        dataset="BOW",
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
                        dataset="BOW",
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
                        dataset="BOW",
                        parameters={"n_jobs": N_JOBS,
                                   "c": c,
                                   "max_iter": LR_ITER})

    report_df = cr.runNewModels()



    log.info("creating TFIDF vector")
    # TODO: play with min_df and max_df
    # TODO: play with variations of ngram
    tv = TfidfVectorizer(min_df=0., max_df=1., ngram_range=(1,3), use_idf=True)
    tv_matrix = tv.fit_transform(X.array)
    vocab = tv.get_feature_names()
    tv_pd = pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=vocab)
    log.info("Finished TFIDF vector")

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




    log.info("Finished running all models")
    end_time = datetime.now()
    total_time = end_time - start_time
    log.info(f'Total time: {total_time}')
    print(report_df.head())

