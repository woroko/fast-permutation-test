# fast-permutation-test
Very fast _exact_ permutation test in Cython. Uses some bit twiddling to generate permutations quickly. Supports multiprocessing with ipyparallel.

A two-tailed test of group means is implemented as an example, but you should be able to substitute your own test statistic.
You probably shouldn't rely on this code unless you have validated the approach independently.
