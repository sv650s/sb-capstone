import util.dict_util as du

class TestDictUtil(object):


    # @pytest.fixture(scope="module")
    # def get_model_failed():
    #     model = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    #     model.predict()

    def test_dict_to_dict(self):
        """
        Test to see if this flattens a dictionary
        :return:
        """
        target = {"a": 0}
        source = {"b": 1, "c": {"1": 2, "2": 3}}
        outcome = du.add_dict_to_dict(target, source)
        key_num = 4
        assert len(outcome.keys()) == key_num, \
            f"number of keys should be {key_num}"


    def test_remove_null(self):
        source = {"a": 1, "b": None, "c": 2}
        outcome = du.remove_null(source)
        assert len(outcome.keys()) == 2, "Number of keys should be 2"
