"""Unit testing
author: steeve laquitaine
"""

from src.nodes.utils import get_vonMises, is_all_in


def test_vonMises():
    pass


def test_is_all_in():
    """unit-test "is_all_in"
    """
    assert (
        is_all_in({0, 1, 2, 3}, {0, 1, 2, 3}) == True
    ), "is_all_in is flawed"

    assert (
        is_all_in({4}, {0, 1, 2, 3}) == False
    ), "is_all_in is flawed"
