import os
import sys
import common
import numpy as np

"""
Stochastic Hopfield model. Write a computer program implementing the Hopfield
model with stochastic updating (LN pp. 36). Use Hebb’s rule and take w ii = 0.
Use random patterns (LN pp. 16 and 19).

a) For β = 2 determine the steady-state order parameter m 1 as a function of
α = p/N for N = 500. Discard data points corresponding to the initial transient
of the stochastic dynamics. Average over independent realisations. Plot the
order parameter as a function of α in the range 0 < α ≤ 1. Discuss your results.
Refer to the phase diagram (LN p. 60).( 1p ).

b) Repeat the above for N = 50 , 100 , and 250 choosing p so that α is in the
same range as above. Discuss how the value of N influences the order parameter m1 .
"""
