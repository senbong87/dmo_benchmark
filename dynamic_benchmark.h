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
const vector<double> LOWER_BOUND(LOWER_ARRAY, LOWER_ARRAY+21);
const vector<double> UPPER_BOUND(UPPER_ARRAY, UPPER_ARRAY+21);

// UTILITY FUNCTION
bool check_boundary(const vector<double> &x)
{
    assert(x.size() == LOWER_BOUND.size() && x.size() == UPPER_BOUND.size());

    bool output = true;
    for (size_t i = 0; i < x.size(); ++i) {
        output = (x[i] >= LOWER_BOUND[i] && x[i] <= UPPER_BOUND[i]);
    } // end for
    return output;
} // end function check_boundary

// DEFINE COMPONENT FUNCTIONS
/*
 * This function is used to calculate the unimodal beta function. Input are the
 * decision variable (x), time (t) and g function (g).
 */
vector<double> beta_uni(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double))
{
    vector<double> beta(2, 0.0);
    for (size_t i = 1; i < x.size(); ++i) {
        if (i % 2) {
            beta[0] += (x[i] - g(x, t))*(x[i] - g(x, t));
        } else {
            beta[1] += (x[i] - g(x, t))*(x[i] - g(x, t));
        } // end if/else
    } // end for
    beta[0] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[0];
    beta[1] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[1];
    return beta;
} // end function beta_uni

/*
 * This function is used to calculate the multi-modal beta function. Input are
 * the decision variable (x), time (t) and g function (g).
 */
vector<double> beta_multi(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double))
{
    double temp;
    vector<double> beta(2, 0.0);
    for (size_t i = 1; i < x.size(); ++i) {
        temp = (x[i] - g(x, t))*(x[i] - g(x, t)) * 
            (1 + fabs(sin(4 * PI * (x[i] - g(x, t)))));
        if (i % 2) {
            beta[0] += temp;
        } else {
            beta[1] += temp;
        } // end if/else
    } // end for
    beta[0] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[0];
    beta[1] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[1];
    return beta;
} // end function beta_multi

/*
 * This function is used to calculate the mixed unimodal and multi-modal beta
 * function. Input are the decision variable (x), time (t) and g function (g).
 */
vector<double> beta_mix(const vector<double> &x, double t,
        double (*g)(const vector<double> &, double))
{
    int k = int(fabs(5*fmod(floor(t/5.0), 2.0) - fmod(t, 5)));
    double temp;
    vector<double> beta(2, 0.0);
    for (size_t i = 1; i < x.size(); ++i) {
        temp = 1 + (x[i] - g(x, t))*(x[i] - g(x, t)) -
            cos(2.0*PI*k*(x[i] - g(x, t)));
        if (i % 2) {
            beta[0] += temp;
        } else {
            beta[1] += temp;
        } // end if/else
    } // end for
    beta[0] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[0];
    beta[1] = 2.0/static_cast<double>(floor(LOWER_BOUND.size()/2.0))*beta[1];
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
    int k = int(fabs(5*fmod(floor(t/5.0), 2.0) - fmod(t, 5)));
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
    int k = int(fabs(5*fmod(floor(t/5.0), 2.0) - fmod(t, 5)));
    double p = log(1-0.1*k)/log(0.1*k+std::numeric_limits<double>::epsilon());
    vector<double> f;
    f.push_back(x[0]);
    f.push_back(1-pow(x[0], p));
    return f;
} // end function alpha_conf

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
    vector<double> f;
    f.push_back(alpha[0] + beta[0]);
    f.push_back(alpha[1] + beta[1]);
    return f;
} // end function additive

/*
 * Multiplicative form of the benchmark problem.
 */
vector<double> multiplicative(const vector<double> &alpha, 
        const vector<double> &beta)
{
    vector<double> f;
    f.push_back(alpha[0]*(1 + beta[0]));
    f.push_back(alpha[1]*(1 + beta[1]));
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

#endif // DYNAMIC_BENCHMARK_H
