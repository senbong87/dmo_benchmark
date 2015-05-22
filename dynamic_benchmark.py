"""Dynamic Multi-objective Optimization Problems

This file contains 12 dynamic benchmark problems which can be used to exaluated
the dynamic multi-objective evolutionary algorithm (DMOEA). Belows are the 
name of the benchmark problems.

	    	DB1a	DB1b	DB2a	DB2b	DB3a	DB3b
    		DB4a	DB4b	DB5a	DB5b	DB6a	DB6b

All the functions have two input arguments: decision variable and time index.
For the time index (t), it is not the generation number used in evolutionary
algorithm.
"""
#!/bin/python

import numpy as np

## Parameter configuration ##
LOWER_BOUND = [0.0] + 20*[-1.0]
UPPER_BOUND = 21*[1.0]
ERR_MSG = "x is outside decision boundary or dimension of x is not correct"
DELTA_STATE = 1


## Define component functions ##
def beta_uni(x, t, g, obj_num=2):
    """This function is used to calculate the unimodal beta function. Input are
    the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0]*obj_num
    for i in range(obj_num-1, len(x)):
        beta[(i+1)%obj_num] += (x[i] - g(x, t))*(x[i] - g(x, t))

    beta = [(2.0/int(len(LOWER_BOUND)/obj_num))*b for b in beta]
    return beta


def beta_multi(x, t, g, obj_num=2):
    """This function is used to calculate the multi-modal beta function. Input 
    are the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0]*obj_num
    for i in range(obj_num-1, len(x)):
        beta[(i+1)%obj_num] += (x[i] - g(x, t))*(x[i] - g(x, t))*\
                (1 + np.abs(np.sin(4*np.pi*(x[i] - g(x, t)))))

    beta = [(2.0/int(len(LOWER_BOUND)/obj_num))*b for b in beta]
    return beta


def beta_mix(x, t, g, obj_num=2):
    """This function is used to calculate the mixed unimodal and multi-modal 
    beta function. Input are the decision variable (x), time (t) and g function
    (g).
    """
    beta = [0.0]*obj_num
    k = int(abs(5.0*(int(DELTA_STATE*int(t)/5.0) % 2) - (DELTA_STATE*int(t) % 5)))

    for i in range(obj_num-1, len(x)):
        temp = 1.0 + (x[i] - g(x, t))*(x[i] - g(x, t)) - np.cos(2*np.pi*k*(x[i] - g(x, t)))
        beta[(i+1)%obj_num] += temp
    beta = [(2.0/int(len(LOWER_BOUND)/obj_num))*b for b in beta]
    return beta


def alpha_conv(x):
    """This function is used to calculate the alpha function with convex POF.
    Input is decision variable (x).
    """
    return [x[0], 1 - np.sqrt(x[0])]


def alpha_disc(x):
    """This function is used to calculate the alpha function with discrete POF.
    Input is decision variable (x).
    """
    return [x[0], 1.5 - np.sqrt(x[0]) - 0.5*np.sin(10*np.pi*x[0])]


def alpha_mix(x, t):
    """This function is used to calculate the alpha function with mixed 
    continuous POF and discrete POF.
    """
    k = int(abs(5.0*(int(DELTA_STATE*int(t)/5.0) % 2) - (DELTA_STATE*int(t) % 5)))
    return [x[0], 1 - np.sqrt(x[0]) + 0.1*k*(1 + np.sin(10*np.pi*x[0]))]


def alpha_conf(x, t):
    """This function is used to calculate the alpha function with time-varying
    conflicting objective. Input are decision variable (x) and time (t).
    """
    k = int(abs(5.0*(int(DELTA_STATE*int(t)/5.0) % 2) - (DELTA_STATE*int(t) % 5)))
    return [x[0], 1 - np.power(x[0], \
            np.log(1 - 0.1*k)/np.log(0.1*k + np.finfo(float).eps))]


def alpha_conf_3obj(x, t):
    """This function is used to calculate the alpha unction with time-varying
    conflicting objective (3-objective). Input are decision variable (x) and 
    time (t).
    """
    k = int(abs(5.0*(int(DELTA_STATE*int(t)/5.0) % 2) - (DELTA_STATE*int(t) % 5)))
    alpha1 = fix_numerical_instability(np.cos(0.5*x[0]*np.pi)*np.cos(0.5*x[1]*np.pi))
    alpha2 = fix_numerical_instability(np.cos(0.5*x[0]*np.pi)*np.sin(0.5*x[1]*np.pi))
    alpha3 = fix_numerical_instability(np.sin(0.5*x[0]*np.pi + 0.25*(k/5.0)*np.pi))
    return [alpha1, alpha2, alpha3]


def g(x, t):
    """This function is used to calculate the g function used in the paper.
    Input are decision variable (x) and time (t). 
    """
    return np.sin(0.5*np.pi*(t-x[0]))


## Utility functions ##
def check_boundary(x, upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND):
    """Check the dimension of x and whether it is in the decision boundary. x is
    decision variable, upper_bound and lower_bound are upperbound and lowerbound
    lists of the decision space
    """
    if len(x) != len(upper_bound) or len(x) != len(lower_bound):
        return False

    output = True
    for e, upp, low in zip(x, upper_bound, lower_bound):
        output = output and (e >= low) and (e <= upp)
    return output


def check_boundary_3obj(x, upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND):
    """Check the dimension of x and whether it is in the decision boundary. x is
    decision variable, upper_bound and lower_bound are upperbound and lowerbound
    lists of the decision space
    """
    lower_bound = [0.0]+lower_bound
    upper_bound = [1.0]+upper_bound
    if len(x) != len(upper_bound) or len(x) != len(lower_bound):
        return False

    output = True
    for e, upp, low in zip(x, upper_bound, lower_bound):
        output = output and (e >= low) and (e <= upp)
    return output


def fix_numerical_instability(x):
    """Check whether x is close to zero, sqrt(0.5) or not. If it is close to 
    these two values, changes x to the value. Otherwise, return x.
    """
    if np.allclose(0.0, x):
        return 0.0
    
    if np.allclose(np.sqrt(0.5), x):
        return np.sqrt(0.5)
    return x


def additive(alpha, beta):
    """Additive form of the benchmark problem.
    """
    return [alpha[0] + beta[0], alpha[1] + beta[1]]


def multiplicative(alpha, beta):
    """Multiplicative form of the benchmark problem.
    """
    return [alpha[0]*(1 + beta[0]), alpha[1]*(1 + beta[1])]


## Benchmark functions ## 
def DB1a(x, t):
    """DB1a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_uni(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB1m(x, t):
    """DB1m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_uni(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB2a(x, t):
    """DB2a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB2m(x, t):
    """DB2m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB3a(x, t):
    """DB3a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB3m(x, t):
    """DB3m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conv(x)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB4a(x, t):
    """DB4a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_disc(x)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB4m(x, t):
    """DB4m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_disc(x)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB5a(x, t):
    """DB5a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB5m(x, t):
    """DB5m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB6a(x, t):
    """DB6a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB6m(x, t):
    """DB6m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_mix(x, t)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB7a(x, t):
    """DB7a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_multi(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB7m(x, t):
    """DB7m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_multi(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB8a(x, t):
    """DB8a dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_mix(x, t, g)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB8m(x, t):
    """DB8m dynamic benchmark problem
    """
    if check_boundary(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf(x, t)
        beta = beta_mix(x, t, g)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB9a(x, t):
    """DB9a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB9m(x, t):
    """DB9m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj(x, t)
        beta = beta_multi(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB10a(x, t):
    """DB10a dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return additive(alpha, beta)
    else:
        raise Exception(ERR_MSG)


def DB10m(x, t):
    """DB10m dynamic benchmark problem
    """
    if check_boundary_3obj(x, UPPER_BOUND, LOWER_BOUND):
        alpha = alpha_conf_3obj(x, t)
        beta = beta_mix(x, t, g, obj_num=3)
        return multiplicative(alpha, beta)
    else:
        raise Exception(ERR_MSG)


if __name__ == '__main__':
    print(__doc__)
