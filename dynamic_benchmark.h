/*
 * dynamic_benchmark.cpp
 *
 * This file contains the dynamic multi-objective optimization benchmark problem
 * used in the paper.
 *
 */
#ifndef DYNAMIC_BENCHMARK_H
#define DYNAMIC_BENCHMARK_H

#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>

using std::vector;
using std::string;

const double PI = 3.14159265359;
const double LOWER_ARRAY[] = { 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
    -1, -1, -1, -1, -1, -1, -1, -1, -1 };
const double UPPER_ARRAY[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1 };
const int DELTA_STATE = 1;
const vector<double> LOWER_BOUND(LOWER_ARRAY, LOWER_ARRAY+21);
const vector<double> UPPER_BOUND(UPPER_ARRAY, UPPER_ARRAY+21);

// UTILITY FUNCTION
bool check_boundary(const vector<double> &x)
{
    assert(x.size() == LOWER_BOUND.size() && x.size() == UPPER_BOUND.size());

    bool output = true;
    for (size_t i = 0; i < x.size(); ++i) {
        output = (x[i] >= LOWER_BOUND[i] && x[i] <= UPPER_BOUND[i]) && output;
    } // end for
    return output;
} // end function check_boundary

bool check_boundary_3obj(const vector<double> &x)
{
    assert(x.size() == LOWER_BOUND.size() + 1);

    bool output = true;
    for (size_t i = 0; i < 2; ++i)
        output = (x[i] >= LOWER_BOUND[0] && x[i] <= UPPER_BOUND[0]) && output;

    for (size_t i = 2; i < x.size(); ++i) {
        output = (x[i] >= LOWER_BOUND[i-1] && x[i] <= UPPER_BOUND[i-1]) && output;
    } // end for
    return output;
} // end function check_boundary_3obj

double fix_numerical_instability(const double x, const double rtol=1e-05, 
        const double atol=1e-08)
{
    if (fabs(x - sqrt(0.5)) < (atol + rtol * sqrt(0.5)))
        return sqrt(0.5);

    if (fabs(x - 0.0) < atol)
        return 0.0;
    return x;
} // end function fix_numerical_instability

// DEFINE COMPONENT FUNCTIONS
/*
 * This function is used to calculate the unimodal beta function. Input are the
 * decision variable (x), time (t) and g function (g).
 */
vector<double> beta_uni(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double), size_t obj_num=2)
{
    vector<double> beta(obj_num, 0.0);
    for (size_t i = obj_num-1; i < x.size(); ++i) {
        beta[(i+1)%obj_num] += (x[i] - g(x, t))*(x[i] - g(x, t));
    } // end for

    for (size_t i = 0; i < beta.size(); ++i)
        beta[i] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/obj_num))*beta[i];
    return beta;
} // end function beta_uni

/*
 * This function is used to calculate the multi-modal beta function. Input are
 * the decision variable (x), time (t) and g function (g).
 */
vector<double> beta_multi(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double), size_t obj_num=2)
{
    double temp;
    vector<double> beta(obj_num, 0.0);
    for (size_t i = obj_num-1; i < x.size(); ++i) {
        beta[(i+1)%obj_num] +=  (x[i] - g(x, t))*(x[i] - g(x, t)) * 
            (1 + fabs(sin(4 * PI * (x[i] - g(x, t)))));
    } // end for
    for (size_t i = 0; i < beta.size(); ++i)
        beta[i] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/obj_num))*beta[i];
    return beta;
} // end function beta_multi

/*
 * This function is used to calculate the mixed unimodal and multi-modal beta
 * function. Input are the decision variable (x), time (t) and g function (g).
 */
vector<double> beta_mix(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double), size_t obj_num=2)
{
    int k = int(fabs(5.0*fmod(floor(DELTA_STATE*int(t)/5.0), 2.0) - fmod(DELTA_STATE*int(t), 5)));
    double temp;
    vector<double> beta(obj_num, 0.0);
    for (size_t i = obj_num-1; i < x.size(); ++i) {
        temp = 1 + (x[i] - g(x, t))*(x[i] - g(x, t)) -
            cos(2.0*PI*k*(x[i] - g(x, t)));
        beta[(i+1)%obj_num] += temp;
    } // end for
    for (size_t i = 0; i < beta.size(); ++i)
        beta[i] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/obj_num))*beta[i];
    return beta;
} // end function beta_mix

/*
 * This function is used to calculate the alpha function with convex PF. Input
 * is decision variable (x). The calculated values are stored in f1 and f2.
 */
vector<double> alpha_conv(const vector<double> &x)
{
    vector<double> f;
    f.push_back(x[0]);
    f.push_back(1 - sqrt(x[0]));
    return f;
} // end function alpha_conv

/*
 * This function is used to calculate the alpha function with discrete PF. Input
 * is decision variable (x). The calculated values are stored in f1 and f2.
 */
vector<double> alpha_disc(const vector<double> &x)
{
    vector<double> f;
    f.push_back(x[0]);
    f.push_back(1.5 - sqrt(x[0]) - 0.5*sin(10.0*PI*x[0]));
    return f;
} // end function alpha_disc

/*
 * This function is used to calculate the alpha function with mix continuous and
 * discrete PF. Input is decision variable (x) and (t). The calculated values are
 * stored in f1 and f2.
 */
vector<double> alpha_mix(const vector<double> &x, double t)
{
    int k = int(fabs(5.0*fmod(floor(DELTA_STATE*int(t)/5.0), 2.0) - fmod(DELTA_STATE*int(t), 5)));
    vector<double> f;
    f.push_back(x[0]);
    f.push_back(1.0 - sqrt(x[0]) + 0.1*k*(1 + sin(10*PI*x[0])));
    return f;
} // end function alpha_mix

/*
 * This function is used to calculate the alpha function with time-varying
 * conflicting objective. Input are decision variables (x) and time (t). The
 * calculated values are stored in f1 and f2.
 */
vector<double> alpha_conf(const vector<double> &x, double t)
{
    int k = int(fabs(5.0*fmod(floor(DELTA_STATE*int(t)/5.0), 2.0) - fmod(DELTA_STATE*int(t), 5)));
    double p = log(1-0.1*k)/log(0.1*k+std::numeric_limits<double>::epsilon());
    vector<double> f;
    f.push_back(x[0]);
    f.push_back(1-pow(x[0], p));
    return f;
} // end function alpha_conf

/*
 * This function is used to calculate the alpha function with time-varying
 * conflicting objective (3-objective, type 1). Input are decision variables (x) 
 * and time (t). The calculated values are stored in f1 and f2.
 */
vector<double> alpha_conf_3obj_type1(const vector<double> &x, double t)
{
    int k = int(fabs(5.0*fmod(floor(DELTA_STATE*int(t)/5.0), 2.0) - fmod(DELTA_STATE*int(t), 5)));
    vector<double> f;
    double alpha1 = fix_numerical_instability(cos(0.5*x[0]*PI)*cos(0.5*x[1]*PI));
    double alpha2 = fix_numerical_instability(cos(0.5*x[0]*PI)*sin(0.5*x[1]*PI));
    double alpha3 = fix_numerical_instability(sin(0.5*x[0]*PI + 0.25*(static_cast<double>(k)/5.0)*PI));
    f.push_back(alpha1);
    f.push_back(alpha2);
    f.push_back(alpha3);
    return f;
} // end function alpha_conf_3obj_type1

/* This function is used to calculate the alpha function with time-varying
 * conflicting objective (3-objective, type 1). Input are decision variables (x) 
 * and time (t). The calculated values are stored in f1 and f2.
 */
vector<double> alpha_conf_3obj_type2(const vector<double> &x, double t)
{
    int k = int(fabs(5.0*fmod(floor(DELTA_STATE*int(t)/5.0), 2.0) - fmod(DELTA_STATE*int(t), 5)));
    double k_ratio = (5.0 - k)/5.0;
    vector<double> f;
    double alpha1 = fix_numerical_instability(cos(0.5*x[0]*PI)*cos(0.5*x[1]*PI*k_ratio));
    double alpha2 = fix_numerical_instability(cos(0.5*x[0]*PI)*sin(0.5*x[1]*PI*k_ratio));
    double alpha3 = fix_numerical_instability(sin(0.5*x[0]*PI));
    f.push_back(alpha1);
    f.push_back(alpha2);
    f.push_back(alpha3);
    return f;
} // end function alpha_conf_3obj_type2

/*
 * This function is used to calculate the g function used in the paper. Input 
 * decision variable (x) and time (t).
 */
double g(const vector<double> &x, double t)
{
    return sin(0.5*PI*(t-x[0]));
} // end function

/*
 * Additive form of the benchmark problem.
 */
vector<double> additive(const vector<double> &alpha, const vector<double> &beta)
{
    assert(alpha.size() == beta.size());
    vector<double> f;
    for (size_t i = 0; i < alpha.size(); ++i)
        f.push_back(alpha[i] + beta[i]);
    return f;
} // end function additive

/*
 * Multiplicative form of the benchmark problem.
 */
vector<double> multiplicative(const vector<double> &alpha, 
        const vector<double> &beta)
{
    assert(alpha.size() == beta.size());
    vector<double> f;
    for (size_t i = 0; i < alpha.size(); ++i)
        f.push_back(alpha[i]*(1 + beta[i]));
    return f;
} // end function multiplicative

/*
 * Benchmark functions
 */
vector<double> DB1a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_uni(x, t, g);
    return additive(alpha, beta);
} // end function DB1a

vector<double> DB1m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_uni(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB1m

vector<double> DB2a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_multi(x, t, g);
    return additive(alpha, beta);
} // end function DB2a

vector<double> DB2m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_multi(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB2m

vector<double> DB3a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_mix(x, t, g);
    return additive(alpha, beta);
} // end function DB3a

vector<double> DB3m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conv(x);
    vector<double> beta = beta_mix(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB3m

vector<double> DB4a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_disc(x);
    vector<double> beta = beta_mix(x, t, g);
    return additive(alpha, beta);
} // end function DB4a

vector<double> DB4m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_disc(x);
    vector<double> beta = beta_mix(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB4m

vector<double> DB5a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_mix(x, t);
    vector<double> beta = beta_multi(x, t, g);
    return additive(alpha, beta);
} // end function DB5a

vector<double> DB5m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_mix(x, t);
    vector<double> beta = beta_multi(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB5m

vector<double> DB6a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_mix(x, t);
    vector<double> beta = beta_mix(x, t, g);
    return additive(alpha, beta);
} // end function DB6a

vector<double> DB6m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_mix(x, t);
    vector<double> beta = beta_mix(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB6m

vector<double> DB7a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conf(x, t);
    vector<double> beta = beta_multi(x, t, g);
    return additive(alpha, beta);
} // end function DB7a

vector<double> DB7m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conf(x, t);
    vector<double> beta = beta_multi(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB7m

vector<double> DB8a(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conf(x, t);
    vector<double> beta = beta_mix(x, t, g);
    return additive(alpha, beta);
} // end function DB8a

vector<double> DB8m(const vector<double> &x, double t)
{
    assert(check_boundary(x));
    vector<double> alpha = alpha_conf(x, t);
    vector<double> beta = beta_mix(x, t, g);
    return multiplicative(alpha, beta);
} // end function DB8m

vector<double> DB9a(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type1(x, t);
    vector<double> beta = beta_multi(x, t, g, 3);
    return additive(alpha, beta);
} // end function DB9a

vector<double> DB9m(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type1(x, t);
    vector<double> beta = beta_multi(x, t, g, 3);
    return multiplicative(alpha, beta);
} // end function DB9m

vector<double> DB10a(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type1(x, t);
    vector<double> beta = beta_mix(x, t, g, 3);
    return additive(alpha, beta);
} // end function DB10a

vector<double> DB10m(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type1(x, t);
    vector<double> beta = beta_mix(x, t, g, 3);
    return multiplicative(alpha, beta);
} // end function DB10m

vector<double> DB11a(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type2(x, t);
    vector<double> beta = beta_multi(x, t, g, 3);
    return additive(alpha, beta);
} // end function DB11a

vector<double> DB11m(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type2(x, t);
    vector<double> beta = beta_multi(x, t, g, 3);
    return multiplicative(alpha, beta);
} // end function DB11m

vector<double> DB12a(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type2(x, t);
    vector<double> beta = beta_mix(x, t, g, 3);
    return additive(alpha, beta);
} // end function DB12a

vector<double> DB12m(const vector<double> &x, double t)
{
    assert(check_boundary_3obj(x));
    vector<double> alpha = alpha_conf_3obj_type2(x, t);
    vector<double> beta = beta_mix(x, t, g, 3);
    return multiplicative(alpha, beta);
} // end function DB12m

#endif // DYNAMIC_BENCHMARK_H
