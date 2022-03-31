# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Copyright 2022 by Steeve Laquitaine, GNU license 
"""


class Model:
    """Abstract model class
    This is the parent class to all model
    It contains generic attributes inherited
    by all models
    """

    def __init__(self):
        pass

    def get_attributes(self):
        """get model's attributes

        Returns:
            (list): list of model attribute
        """
        return [
            k
            for k, v in vars(self).items()
            if not (k.startswith("_") or callable(v))
        ]
