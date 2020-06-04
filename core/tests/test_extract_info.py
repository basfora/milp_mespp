import pytest
from core import extract_info as ext


def test_get_idx_searchers():
    info_vector = [5, 2]
    assert ext.get_idx_searchers(info_vector)[0] == [0, 1]
    assert ext.get_idx_searchers(info_vector)[1] == 2


def test_get_set_searchers():
    info_vector = [5, 2]
    assert ext.get_set_searchers(info_vector)[0] == [1, 2]
    assert ext.get_set_searchers(info_vector)[1] == 2

    searchers_info = {1: 2, 2: 3}
    S, m = ext.get_set_searchers(searchers_info)

    assert S == [1, 2]
    assert m == 2

