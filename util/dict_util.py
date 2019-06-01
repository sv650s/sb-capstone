
# expand classification report into dictionary
# classifcation report is a 2 level dictionary. from documentation, it looks something like this
# {'label 1': {'precision':0.5,
#              'recall':1.0,
#              'f1-score':0.67,
#              'support':1},
#  'label 2': { ... },
#   ...
# }
# resulting dictionary:
#   label1_precision: 0.5,
#   label1_recall: 1.0
#   label1_f1-score: 0.67
#
# TODO: should change this to recursively flatten dictionary
def add_dict_to_dict(target :dict, source :dict) -> dict:
    """
    target: dictionary to add to
    source: dictionary to add from
    ------
    return: dictionary with source added to target
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # append key to dictionary keys
            for subkey, subvalue in value.items():
                target[f'{key}_{subkey}'] = subvalue
        else:
            target[key] = value

    return target
