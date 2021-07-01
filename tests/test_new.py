from ell import *
import numpy as np

def test_new():
    a=Ell1d.zero(min_index=0, max_index=10)
    a=Ell2d(array=[[1,2,3,4], [2,2,2,2]], min_index=0)
    try:
        Ell1d(array=[[1,2,3,4], [2,2,2,2]], min_index=0)
    except DimError as d:
        print('I catch the dim error!')
    else:
        raise Exception('I loss the dim error!')


test_new()
