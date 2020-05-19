import pytest
from util.model_util import Loader, ModelCache
from util.tf2_util import AttentionLayer
import inspect


class TestLoader(object):

    @pytest.fixture(scope="class")
    def custom_object_dict(self):
        return { "AttentionLayer": "util.tf2_util.AttentionLayer" }


    def test_loader_get_custom_objects(self, custom_object_dict):
        """
        Convert str dictionary to custom_object dictionary
        1. check to see if key is correctly loaded
        2. check to see if value is a class object
        :param custom_object_dict:
        :return:
        """
        ret = Loader.get_custom_objects(custom_object_dict)

        key = "AttentionLayer"
        assert list(ret.keys())[0] == key, f"key should be {key}"
        assert inspect.isclass(ret[key]), f"value should be {key}"



class TestModelCache(object):

    def test_put(self):
        mc = ModelCache()
        mc.put("hi")
        assert mc.size() == 1, "size should be 1"