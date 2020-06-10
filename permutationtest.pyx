from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.view cimport array as cvarray
import numpy as np
import cython
from libc.math cimport fabs
from functools import reduce
import operator as op

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef inline double mean(double *input_arr, int length):
    cdef double sum = 0.0
    cdef int i
    for i in range(length):
        sum += input_arr[i]
    
    return sum/length

# returns raw bitmasks
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def raw_gosper_bitmask(int A_len, long long K, long long work_divider):
    #subsets = []
    cdef long long num_combinations = 0
    cdef long long N = A_len
    cdef int N_int = A_len

    # iterate over subsets of size K
    cdef long long mask = ((<long long>1)<<K)-1     # 2^K - 1 is always a number having exactly K 1 bits
    
    cdef long long a, b
    cdef int n
    cdef long long permutations_per_job = ncr(A_len, K) // work_divider
    split_masks = []
    
    while mask < ((<long long>1)<<N):
        if mask == 0:
            break
        
        if num_combinations % permutations_per_job == 0:
            split_masks.append(mask)
        # determine next mask with Gosper's hack
        a = mask & -mask                # determine rightmost 1 bit
        b = mask + a                    # determine carry bit
        mask = <long long>(((mask^b)>>2)/a) | b # produce block of ones that begins at the least-significant bit
        #mask = b +(((b^mask)/a)>>2) # wikipedia version, no difference to above
        num_combinations += 1
    

    return (split_masks, permutations_per_job, num_combinations)


# two-sided permutation test of group means
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def permutation_test_parallel(double[:] A, long long K, double given_mean, long long mask_start, long long mask_end):
    #subsets = []
    cdef long long num_combinations = 0
    cdef long long N = A.size
    cdef int N_int = A.size
    #subset = [0] * K
    cdef int subset_size = <int>K
    cdef int subset_remainder_size = <int>(N-K)
    cdef double *subset = <double *> PyMem_Malloc(subset_size * sizeof(double))
    cdef double *subset_remainder = <double *> PyMem_Malloc(subset_remainder_size * sizeof(double))
    
    cdef int subset_index = 0
    cdef int subset_remainder_index = 0

    # iterate over subsets of size K
    cdef long long mask = mask_start
    #((<long long>1)<<K)-1     # 2^K - 1 is always a number having exactly K 1 bits
    
    cdef long long a, b
    cdef int n
    
    # two-tailed
    cdef double teststatistic_difference = fabs(given_mean)
    cdef long long extreme_teststatistic_count = 0
    
    while mask < mask_end:
        subset_index = 0
        subset_remainder_index = 0
        for n in range(N_int):
            if ((mask>>n)&1) == 1:
                subset[subset_index] = A[n]
                subset_index += 1
            else:
                subset_remainder[subset_remainder_index] = A[n]
                subset_remainder_index += 1
        
        #subsets.append(subset)
 
        # catch special case
        if mask == 0:
            break
        
        
        # insert your own test
        # currently does two-tailed test of means
        if fabs(mean(subset, subset_size) - mean(subset_remainder, subset_remainder_size)) >= teststatistic_difference:
            extreme_teststatistic_count += 1
        
 
        # determine next mask with Gosper's hack
        a = mask & -mask                # determine rightmost 1 bit
        b = mask + a                    # determine carry bit
        mask = <long long>(((mask^b)>>2)/a) | b # produce block of ones that begins at the least-significant bit
        
        num_combinations += 1
    
    PyMem_Free(subset)
    PyMem_Free(subset_remainder)

    return (extreme_teststatistic_count, num_combinations)


# two-sided permutation test of group means
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def permutation_test(double[:] A, long long K, double given_mean):
    #subsets = []
    cdef long long num_combinations = 0
    cdef long long N = A.size
    cdef int N_int = A.size
    #subset = [0] * K
    cdef int subset_size = <int>K
    cdef int subset_remainder_size = <int>(N-K)
    cdef double *subset = <double *> PyMem_Malloc(subset_size * sizeof(double))
    cdef double *subset_remainder = <double *> PyMem_Malloc(subset_remainder_size * sizeof(double))
    
    cdef int subset_index = 0
    cdef int subset_remainder_index = 0

    # iterate over subsets of size K
    cdef long long mask = ((<long long>1)<<K)-1     # 2^K - 1 is always a number having exactly K 1 bits
    
    cdef long long a, b
    cdef int n
    
    # two-tailed
    cdef double teststatistic_difference = fabs(given_mean)
    cdef long long extreme_teststatistic_count = 0
    
    while mask < ((<long long>1)<<N):
        subset_index = 0
        subset_remainder_index = 0
        for n in range(N_int):
            if ((mask>>n)&1) == 1:
                subset[subset_index] = A[n]
                subset_index += 1
            else:
                subset_remainder[subset_remainder_index] = A[n]
                subset_remainder_index += 1
        
        #subsets.append(subset)
 
        # catch special case
        if mask == 0:
            break
        
        
        # insert your own test
        # currently does two-tailed test of means
        if fabs(mean(subset, subset_size) - mean(subset_remainder, subset_remainder_size)) >= teststatistic_difference:
            extreme_teststatistic_count += 1
        
 
        # determine next mask with Gosper's hack
        a = mask & -mask                # determine rightmost 1 bit
        b = mask + a                    # determine carry bit
        mask = <long long>(((mask^b)>>2)/a) | b # produce block of ones that begins at the least-significant bit
        
        num_combinations += 1
    
    PyMem_Free(subset)
    PyMem_Free(subset_remainder)

    return (extreme_teststatistic_count, num_combinations)