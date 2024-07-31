# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:14:44 2024

@author: rigved
"""

import numpy as np
import itertools

def s(x):
  return 1/(1 + np.exp(-x))

################################
# Task 1.2
################################


# f gate
w_fx = -2
w_fh = -2
b_f = -2

# i gate
w_ix = 10
w_ih = 10
b_i = -5

# g
w_gx = 2
w_gh = 2
b_g = 2

# o gate
w_ox = -6
w_oh = -6
b_o = 10

################################

# The below code runs through all length 14 binary strings and throws an error 
# if the LSTM fails to predict the correct parity

cnt = 0
for X in itertools.product([0,1], repeat=14):
  c=0
  h=0
  cnt += 1
  for x in X:
    i = s(w_ih*h + w_ix*x + b_i)
    f = s(w_fh*h + w_fx*x + b_f)
    g = np.tanh(w_gh*h + w_gx*x + b_g)
    o = s(w_oh*h + w_ox*x + b_o)
    c = f*c + i*g
    h = o*np.tanh(c)
  if np.sum(X)%2 != int(h>0.5):
    print("Failure",cnt, X, int(h>0.5), np.sum(X)%2 == int(h>0.5))
    break
  if cnt % 1000 == 0:
    print(cnt)