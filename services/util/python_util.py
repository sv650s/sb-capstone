import importlib
import logging


logger = logging.getLogger(__name__)


def load_class(classpath: str):
    logger.info(f"creating class from {classpath}")
    return getattr(importlib.import_module(".".join(classpath.split(".")[:-1])), classpath.split(".")[-1])


def load_instance(classpath: str):
    logger.info(f"creating instance of class from {classpath}")
    class_ = load_class(classpath)
    return class_()
