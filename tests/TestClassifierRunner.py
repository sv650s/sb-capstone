from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from models.ClassifierRunner import ClassifierRunner, Keys, Status
from pandas import DataFrame
from pprint import pprint
import logging
import pytest

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s - %(message)s', level=logging.DEBUG)

class TestClassiferRunner(object):

    @pytest.fixture(scope="module")
    def get_train_x(self):
        train_x = DataFrame()
        train_x = train_x.append({"a":1, "b":1}, ignore_index=True)
        train_x = train_x.append({"a":2, "b":2}, ignore_index=True)
        return train_x

    @pytest.fixture(scope="module")
    def get_train_y(self):
        return DataFrame([1, 2])

    def test_add_model(self, get_train_x, get_train_y):
        train_x = get_train_x
        train_y = get_train_y

        neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        cr = ClassifierRunner(write_to_csv=False)
        cr.addModel(neigh, train_x, train_y, train_x, train_y, name="test_success")
        models = cr.models
        assert len(models) == 1, "length of models should be 1"

        for model in models:
            print("models:\n--------")
            pprint(model)



    def test_run_one_model(self):
        """
        :return:
        """

        # train_x = DataFrame()
        # train_x = train_x.append({"a":1, "b":1}, ignore_index=True)
        # train_x = train_x.append({"a":2, "b":2}, ignore_index=True)
        # train_y = DataFrame([1, 2])
        # neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        #
        # cr = ClassifierRunner(write_to_csv=False)
        # cr.addModel(neigh, train_x, train_y, train_x, train_y, name="success")
        # report_df = cr.runModels()
        #
        # assert len(report_df) == 1, "report should have 1 entry"
        # assert report_df.iloc[0, Keys.STATUS] == Status.SUCCESS, "status should be SUCCESS"



    def test_run_two_model(self):
        """
        :return:
        """


    def test_dict_to_dict(self):
        """
        Test to see if this flattens a dictionary
        :return:
        """
        target = {"a": 0}
        source = {"b": 1, "c": {"1": 2, "2": 3}}
        outcome = ClassifierRunner._add_dict_to_dict(target, source)
        key_num = 4
        assert len(outcome.keys()) == key_num, \
            f"number of keys should be {key_num}"

