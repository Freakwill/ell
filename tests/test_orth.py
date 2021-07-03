#!/usr/bin/env python3

from ell import *

_filter = Filter.from_name('db4')
_filter.orth_test()
one=Ell1d.unit()

a= Ell1d([0,0,1,1,0,0])

_filter.tensor().orth_test()

a= Ell1d([0,0,1,1,0,0]).tensor()