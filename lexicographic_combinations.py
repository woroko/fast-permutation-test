#! /usr/bin/python
#
# This snippet provides a simple function "combination" that can compute an
# arbitrary k-combination of n elements given an index m into the
# lexicographically ordered set of all k-combinations of n elements.


from math import factorial


# Compute the total number of unique k-combinations in a set of n elements.
# There are more efficient implementations of the choose function, but
# that's not the main point of this snippet.
def choose(n, k):
    if n < k:
        return 0
    return factorial(n) / (factorial(k) * factorial(n - k))


# Compute the mth combination in lexicographical order from a set of n
# elements chosen k at a time.
# Algorithm from http://msdn.microsoft.com/en-us/library/aa289166(v=vs.71).aspx
def nth_combination(n, k, m):
    result = []
    a      = n
    b      = k
    x      = (choose(n, k) - 1) - m
    for i in range(0, k):
        a = a - 1
        while choose(a, b) > x:
            a = a - 1
        result.append(n - 1 - a)
        x = x - choose(a, b)
        b = b - 1
    return result