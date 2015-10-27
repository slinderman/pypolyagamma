// -*- c-basic-offset: 4; -*-
#include "RNG.hpp"

// #ifdef USE_R
// #include "RRNG.cpp"
// #else
// #include "GRNG.cpp"
// #endif

// Throw runtime exception or return.
#ifndef TREOR
#ifndef NTHROW
#define TREOR(MESS, VAL) throw std::runtime_error(MESS);
#else
#define TREOR(MESS, VAL) {fprintf(stderr, MESS); return VAL;}
#endif
#endif

#ifndef RCHECK
#define RCHECK 1000
#endif

RNG::RNG(unsigned long seed) : BasicRNG(seed) {}

inline void check_R_interupt(int count)
{
    #ifdef USE_R
    if (count % RCHECK == 0) R_CheckUserInterrupt();
    #endif
}

// Truncated Exponential
double RNG::texpon_rate(double left, double rate){
    if (rate < 0) TREOR("texpon_rate: rate < 0, return 0\n", 0.0);
    // return left - log(unif()) / rate;
    return expon_rate(rate) + left;
}

double RNG::texpon_rate(double left, double right, double rate)
{
    if (left == right) return left;
    if (left > right) TREOR("texpon_rate: left > right, return 0.\n", 0.0);
    if (rate < 0) TREOR("texpon_rate: rate < 0, return 0\n", 0.0);

    double b = 1 - exp(rate * (left - right));
    double y = 1 - b * unif();
    return left - log(y) / rate;
}

//////////////////////////////////////////////////////////////////////
	       // TRUNCATED NORMAL HELPER FUNCTIONS //
//////////////////////////////////////////////////////////////////////

double RNG::alphastar(double left)
{
    return 0.5 * (left + sqrt(left*left + 4));
} // alphastar

double RNG::lowerbound(double left)
{
    double astar  = alphastar(left);
    double lbound = left + exp(0.5 * + 0.5 * left * (left - astar)) / astar;
    return lbound;
} // lowerbound

//////////////////////////////////////////////////////////////////////
		     // DRAW TRUNCATED NORMAL //
//////////////////////////////////////////////////////////////////////

double RNG::tnorm(double left)
{
    double rho, ppsl;
    int count = 1;

    if (left < 0) { // Accept/Reject Normal
        while (true) {
            ppsl = norm(0.0, 1.0);
            if (ppsl > left) return ppsl;
            check_R_interupt(count++);
            #ifndef NDEBUG
            if (count > RCHECK * 1000) fprintf(stderr, "left < 0; count: %i\n", count);
            #endif
        }
    }
    else { // Accept/Reject Exponential
        // return tnorm_tail(left); // Use Devroye.
        double astar = alphastar(left);
        while (true) {
            ppsl = texpon_rate(left, astar);
            rho  = exp( -0.5 * (ppsl - astar) * (ppsl - astar) );
            if (unif() < rho) return ppsl;
            check_R_interupt(count++);
            #ifndef NDEBUG
            if (count > RCHECK * 1000) fprintf(stderr, "left > 0; count: %i\n", count);
            #endif
        }
    }
} // tnorm
//--------------------------------------------------------------------

double RNG::tnorm(double left, double right)
{
    // The most difficult part of this algorithm is figuring out all the
    // various cases.  An outline is summarized in the Appendix.

    // Check input
    #ifdef USE_R
    if (ISNAN(right) || ISNAN(left))
    #else
    if (std::isnan(right) || std::isnan(left))
    #endif
	{
	    fprintf(stderr, "Warning: nan sent to RNG::tnorm: left=%g, right=%g\n", left, right);
	    TREOR("RNG::tnorm: parameter problem.\n", 0.5 * (left + right));
	    // throw std::runtime_error("RNG::tnorm: parameter problem.\n");
	}
    
    if (right < left) {
        fprintf(stderr, "Warning: left: %g, right:%g.\n", left, right);
        TREOR("RNG::tnorm: parameter problem.\n", 0.5 * (left + right));
    }
    
    double rho, ppsl;
    int count = 1;
    
    if (left >= 0) {
        double lbound = lowerbound(left);
        if (right > lbound) { // Truncated Exponential.
            double astar = alphastar(left);
            while (true) {
		ppsl = texpon_rate(left, right, astar);
                rho  = exp(-0.5*(ppsl - astar)*(ppsl-astar));
                if (unif() < rho) return ppsl;
		if (count > RCHECK * 10) fprintf(stderr, "left >= 0, right > lbound; count: %i\n", count);
                // if (ppsl < right) return ppsl;
            }
        }
        else {
            while (true) {
                ppsl = flat(left, right);
                rho  = exp(0.5 * (left*left - ppsl*ppsl));
                if (unif() < rho) return ppsl;
                check_R_interupt(count++);
                #ifndef NDEBUG
                if (count > RCHECK * 10) fprintf(stderr, "left >= 0, right <= lbound; count: %i\n", count);
                #endif
            }
        }
    }
    else if (right >= 0) {
        if ( (right - left) < SQRT2PI ){
            while (true) {
                ppsl = flat(left, right);
                rho  = exp(-0.5 * ppsl * ppsl);
                if (unif() < rho) return ppsl;
                check_R_interupt(count++);
                #ifndef NDEBUG
                if (count > RCHECK * 10) fprintf(stderr, "First, left < 0, right >= 0, count: %i\n", count);
                #endif
            }
        }
        else{
            while (true) {
                ppsl = norm(0, 1);
                if (left < ppsl && ppsl < right) return ppsl;
                check_R_interupt(count++);
                #ifndef NDEBUG
                if (count > RCHECK * 10) fprintf(stderr, "Second, left < 0, right > 0, count: %i\n", count);
                #endif
            }
        }
    }
    else {
        return -1. * tnorm(-1.0 * right, -1.0 * left);
    }
} // tnorm
//--------------------------------------------------------------------

double RNG::tnorm(double left, double mu, double sd)
{
    double newleft = (left - mu) / sd;
    return mu + tnorm(newleft) * sd;
} // tnorm
//--------------------------------------------------------------------

double RNG::tnorm(double left, double right, double mu, double sd)
{
    if (left==right) return left;

    double newleft  = (left - mu) / sd;
    double newright = (right - mu) / sd;

    // I want to check this here as well so we can see what the input was.
    // It may be more elegant to try and catch tdraw.
    if (newright < newleft) {
        fprintf(stderr, "left, right, mu, sd: %g, %g, %g, %g \n", left, right, mu, sd);
        fprintf(stderr, "nleft, nright: %g, %g\n", newleft, newright);
        TREOR("RNG::tnorm: parameter problem.\n", 0.5 * (left + right));
    }

    double tdraw = tnorm(newleft, newright);
    double draw = mu + tdraw * sd;

    // It may be the case that there is some numerical error and that the draw
    // ends up out of bounds.
    if (draw < left || draw > right){
        fprintf(stderr, "Error in tnorm: draw not in bounds.\n");
        fprintf(stderr, "left, right, mu, sd: %g, %g, %g, %g\n", left, right, mu, sd);
        fprintf(stderr, "nleft, nright, tdraw, draw: %g, %g, %g, %g\n", newleft, newright, tdraw, draw);
        TREOR("Aborting and returning average of left and right.\n",  0.5 * (left + right));
    }

    return draw;
} // tnorm
//--------------------------------------------------------------------

// Right tail of normal by Devroye
//------------------------------------------------------------------------------
double RNG::tnorm_tail(double t)
{
    int count = 1;

    double E1 = expon_rate(1.0);
    double E2 = expon_rate(1.0);
    while ( E1*E1 > 2 * E2 / t) {
        E1 = expon_rate(1.0);
        E2 = expon_rate(1.0);
        check_R_interupt(count++);
        if (count > RCHECK * 1000) fprintf(stderr, "tnorm_tail; count: %i\n", count);
    }
    return (1 + t * E1) / sqrt(t);
}

//------------------------------------------------------------------------------

// Truncatation at t = 1.
inline double RNG::right_tgamma_reject(double shape, double rate)
{
    double x = 2.0;
    while (x > 1.0)
        x = gamma_rate(shape, rate);
    return x;
}

double RNG::omega_k(int k, double a, double b)
{
    double log_coef = -b + (a+k-1) * log(b) - Gamma(a+k, true) - p_gamma_rate(1.0, a, b, true);
    return exp(log_coef);
}

// Truncation at t = 1.
double RNG::right_tgamma_beta(double shape, double rate)
{
    double a = shape;
    double b = rate;

    double u = unif();

    int k = 1;
    double cdf = omega_k(1, a, b);
    while (u > cdf) {
        cdf += omega_k(++k, a, b);
        if (k % 100000 == 0) {
            printf("right_tgamma_beta (itr k=%i): a=%g, b=%g, u=%g, cdf=%g\n", k, a, b, u, cdf);
#ifdef USE_R
            R_CheckUserInterrupt();
#endif
        }
    }

    return beta(a, k);
}

double RNG::rtgamma_rate(double shape, double rate, double right_t)
{
    // x \sim (a,b,t)
    // ty = x
    // y \sim (a, bt, 1);
    double a = shape;
    double b = rate * right_t;

    double p = p_gamma_rate(1, a, b);
    double y = 0.0;
    if (p > 0.95)
        y = right_tgamma_reject(a, b);
    else
        y = right_tgamma_beta(a,b);

    double x = right_t * y;
    return x;
}

//------------------------------------------------------------------------------
double RNG::ltgamma(double shape, double rate, double trunc)
{
    double a = shape;
    double b = rate * trunc;

    if (trunc <=0) {
        fprintf(stderr, "ltgamma: trunc = %g < 0\n", trunc);
        return 0;
    }
    if (shape < 1) {
        fprintf(stderr, "ltgamma: shape = %g < 1\n", shape);
        return 0;
    }

    if (shape ==1) return expon_rate(1) / rate + trunc;

    double d1 = b-a;
    double d3 = a-1;
    double c0 = 0.5 * (d1 + sqrt(d1*d1 + 4 * b)) / b;

    double x = 0.0;
    bool accept = false;

    while (!accept) {
        x = b + expon_rate(1) / c0;
        double u = unif();

        double l_rho = d3 * log(x) - x * (1-c0);
        double l_M   = d3 * log(d3 / (1-c0)) - d3;

        accept = log(u) <= (l_rho - l_M);
    }

    return trunc * (x/b);
}

//------------------------------------------------------------------------------
double RNG::igauss(double mu, double lambda)
{
    // See R code for specifics.
    double mu2 = mu * mu;
    double Y = norm(0.0, 1.0);
    Y *= Y;
    double W = mu + 0.5 * mu2 * Y / lambda;
    double X = W - sqrt(W*W - mu2);
    if (unif() > mu / (mu + X))
        X = mu2 / X;
    return X;
}

//------------------------------------------------------------------------------
double RNG::rtinvchi2(double scale, double trunc)
{
    double R = trunc / scale;
    // double X = 0.0;
    // // I need to consider using a different truncated normal sampler.
    // double E1 = r.expon_rate(1.0); double E2 = r.expon_rate(1.0);
    // while ( (E1*E1) > (2 * E2 / R)) {
    //   // printf("E %g %g %g %g\n", E1, E2, E1*E1, 2*E2/R);
    //   E1 = r.expon_rate(1.0); E2 = r.expon_rate(1.0);
    // }
    // // printf("E %g %g \n", E1, E2);
    // X = 1 + E1 * R;
    // X = R / (X * X);
    // X = scale * X;
    double E = tnorm(1/sqrt(R));
    double X = scale / (E*E);
    return X;
}

//------------------------------------------------------------------------------

double RNG::Beta(double a, double b, bool log)
{
    double out = Gamma(a, true) + Gamma(b, true) - Gamma(a+b,true);
    if (!log) out = exp(out);
    return out;
}

//------------------------------------------------------------------------------

double RNG::p_igauss(double x, double mu, double lambda)
{
    // z = 1 / mean
    double z = 1 / mu;
    double b = sqrt(lambda / x) * (x * z - 1);
    double a = sqrt(lambda / x) * (x * z + 1) * -1.0;
    double y = RNG::p_norm(b) + exp(2 * lambda * z) * RNG::p_norm(a);
    return y;
}
