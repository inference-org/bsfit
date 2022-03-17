# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""    
    Unit-testing module

    usage:

    .. highlight:: python
    .. code-block:: python
    
        python test.py

    :copyright: Copyright 2022 by Steeve Laquitaine, see AUTHORS.
    :license: ???, see LICENSE for details.
"""


import numpy as np

from src.nodes.data import VonMises
from src.nodes.util import is_all_in


def test_VonMises():
    """test VonMises data class
    """
    vmises = VonMises(p=True).get(
        v_x=np.arange(1, 361, 1),
        v_u=np.arange(1, 361, 1),
        v_k=[0.5, 1],
    )
    # check shape
    assert vmises.shape == (
        360,
        720,
    ), "measure density's shape is wrong"

    # check normalization
    assert all(
        sum(vmises)
    ), "VonMises are not probabilities"


def test_is_all_in():
    """test "is_all_in" function
    """
    assert (
        is_all_in({0, 1, 2, 3}, {0, 1, 2, 3}) == True
    ), "is_all_in is flawed"

    assert (
        is_all_in({4}, {0, 1, 2, 3}) == False
    ), "is_all_in is flawed"
