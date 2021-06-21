#!/usr/bin/env python3

from ell import *


d_filter.orth_test()

a= Ell1D([0,0,1,1,0,0])
print(a.filter(d_filter) + a.filter(d_filter.check()))

d_filter.tensor().orth_test()

a= Ell1D([0,0,1,1,0,0]).tensor()
print(a.filter(d_filter) + a.filter(d_filter.check()))