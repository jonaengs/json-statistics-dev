import math
from itertools import starmap

def my_error(truth, estimate):
    diff = abs(truth - estimate)
    error =  diff / (min(truth, estimate) or 1)
    return error
    # return error - 1

def absolute_error(truth, estimate):
    return abs(truth - estimate)

def relative_error(truth, estimate):
    return abs(truth - estimate) / (truth or 1)

def symmetric_relative_error(truth, estimate):
    if truth == estimate == 0: return 0
    return abs(truth - estimate) / ((truth + estimate) / 2)

def symmetric_relative_error_alt(truth, estimate):
    if truth == estimate == 0: return 0
    return abs(truth - estimate) / (truth + estimate)

def log_relative_error(truth, estimate):
    if estimate == 0: 
        estimate += 1
        truth += 1

    return math.log10(estimate / (truth or 1))

def log_absolute_error(truth, estimate):
    return math.log10(abs(truth - estimate) or 1)

def squared_error(truth, estimate):
    return (truth - estimate)**2


########## AGGREGATE FUNCTIONS ##########

def mean_my_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))
    
    summed = sum(starmap(my_error, zip(truth_arr, estimate_arr)))
    return summed / len(truth_arr)

def mean_log_relative_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))
    
    summed = sum(starmap(log_relative_error, zip(truth_arr, estimate_arr)))
    return summed / len(truth_arr)

def mean_abs_log_relative_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))
    
    summed = sum(map(abs, starmap(log_relative_error, zip(truth_arr, estimate_arr))))
    return summed / len(truth_arr)


# MSE (aka MSD)
def mean_squared_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))

    summed_squared = sum(starmap(squared_error, zip(truth_arr, estimate_arr)))
    return summed_squared / len(truth_arr)

# RMSE (aka RMSD)
def root_mean_squared_error(truth_arr, estimate_arr):
    return math.sqrt(mean_squared_error(truth_arr, estimate_arr))

# MAPE
def mean_absolute_percent_error(truth_arr, estimate_arr):
    return 100 * mean_absolute_percent_error(truth_arr, estimate_arr)

# MAPE no percent ("MARE"?)
def mean_absolute_relative_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))

    summed_relative = sum(starmap(relative_error, zip(truth_arr, estimate_arr)))
    return summed_relative / len(truth_arr)

# SMAPE
def symmetric_mean_absolute_percent_error(truth_arr, estimate_arr):
    return 100 * symmetric_mean_absolute_relative_error(truth_arr, estimate_arr)

# SMAPE no percent ("SMARE"?)
def symmetric_mean_absolute_relative_error(truth_arr, estimate_arr):
    assert(len(truth_arr) == len(estimate_arr))
    
    summed_symmetric_relative = sum(starmap(symmetric_relative_error, zip(truth_arr, estimate_arr)))
    return summed_symmetric_relative / len(truth_arr)

# SMAPE alternative with range 0%-100%
def symmetric_mean_absolute_percent_error_alt(truth_arr, estimate_arr):
    return 100 * symmetric_mean_absolute_relative_error_alt(truth_arr, estimate_arr)

# SMAPE no percent alternative in [0, 1]
def symmetric_mean_absolute_relative_error_alt(truth_arr, estimate_arr):
    return symmetric_mean_absolute_relative_error(truth_arr, estimate_arr) / 2


