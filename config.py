import yaml
import json
from types import SimpleNamespace
import constants


def object_hook(d):
    return SimpleNamespace(**d)


with open(constants.CONFIG_PATH, "r") as yaml_file:
    config_usual_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

# converts dictionary to object:
# https://www.kite.com/python/answers/how-to-convert-a-dictionary-into-an-object-in-python
# https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object/15882054#15882054
config_dump = json.dumps(config_usual_dict)
config: object = json.loads(config_dump, object_hook=object_hook)
