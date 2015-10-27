// -*- mode: c++; -*-

////////////////////////////////////////////////////////////////////////////////

// Copyright 2012 Nick Polson, James Scott, and Jesse Windle.

// This file is part of BayesLogit.

// BayesLogit is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.

// BayesLogit is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with
// BayesLogit.  If not, see <http://www.gnu.org/licenses/>.

////////////////////////////////////////////////////////////////////////////////

// See <http://arxiv.org/abs/1205.0310> for implementation details.

#ifndef __POLYAGAMMAPAR__
#define __POLYAGAMMAPAR__

#include "RNG.hpp"
// #include "Matrix.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <omp.h>

#include "PolyaGamma.h"

// // The numerical accuracy of __PI will affect your distribution.
// const double __PI = 3.141592653589793238462643383279502884197;
// const double HALFPISQ = 0.5 * __PI * __PI;
// const double FOURPISQ = 4 * __PI * __PI;
// const double __TRUNC = 0.64;
// const double __TRUNC_RECIP = 1.0 / __TRUNC;

class PolyaGammaPar
{

  // For sum of Gammas.
  int T;
  std::vector<double> b;

 public:

  // Constructors.
  PolyaGammaPar(int trunc = 200);

  // Draw.
  // double draw(double n, double z, RNG& r);
  double draw(int n, double z, RNG& r);
  double draw_sum_of_gammas(double n, double z, RNG& r);
  double draw_like_devroye(double z, RNG& r);
  void draw(double* samp, int* n, double* z, const int N, const int nthreads);

  //void draw(MF x, double a, double z, RNG& r);
  //void draw(MF x, MF a, MF z, RNG& r);

  // Utility.
  void set_trunc(int trunc);

  // Helper.
  double a(int n, double x);
  double pigauss(double x, double Z);
  double mass_texpon(double Z);
  double rtigauss(double Z, RNG& r);

};

////////////////////////////////////////////////////////////////////////////////
			       // Constructors //
////////////////////////////////////////////////////////////////////////////////

PolyaGammaPar::PolyaGammaPar(int trunc) 
  : T(trunc)
  , b(T)
{
  set_trunc(T);
} // PolyaGammaPar

////////////////////////////////////////////////////////////////////////////////
				 // Utility //
////////////////////////////////////////////////////////////////////////////////

void PolyaGammaPar::set_trunc(int trunc)
{
  if (trunc < 1)
    throw std::invalid_argument("PolyaGammaPar(int trunc): trunc < 1.");

  T = trunc;
  b.resize(T);

  for(int k=0; k < T; ++k){
    // + since we start indexing at 0.
    double d = ((double) k + 0.5);
    b[k] = FOURPISQ * d * d;
  }
} // set_trunc

double PolyaGammaPar::a(int n, double x)
{
  double K = (n + 0.5) * __PI;
  double y = 0;
  if (x > __TRUNC) {
    y = K * exp( -0.5 * K*K * x );
  }
  else if (x > 0) {
    double expnt = -1.5 * (log(0.5 * __PI)  + log(x)) + log(K) - 2.0 * (n+0.5)*(n+0.5) / x;
    y = exp(expnt);
    // y = pow(0.5 * __PI * x, -1.5) * K * exp( -2.0 * (n+0.5)*(n+0.5) / x);
    // ^- unstable for small x?
  }
  return y;
}

double PolyaGammaPar::pigauss(double x, double Z)
{
  double b = sqrt(1.0 / x) * (x * Z - 1);
  double a = sqrt(1.0 / x) * (x * Z + 1) * -1.0;
  double y = RNG::p_norm(b) + exp(2 * Z) * RNG::p_norm(a);
  return y;
}

double PolyaGammaPar::mass_texpon(double Z)
{
  double t = __TRUNC;

  double fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  double b = sqrt(1.0 / t) * (t * Z - 1);
  double a = sqrt(1.0 / t) * (t * Z + 1) * -1.0;

  double x0 = log(fz) + fz * t;
  double xb = x0 - Z + RNG::p_norm(b, 1);
  double xa = x0 + Z + RNG::p_norm(a, 1);

  double qdivp = 4 / __PI * ( exp(xb) + exp(xa) );

  return 1.0 / (1.0 + qdivp);
}

double PolyaGammaPar::rtigauss(double Z, RNG& r)
{
  Z = fabs(Z);
  double t = __TRUNC;
  double X = t + 1.0;
  if (__TRUNC_RECIP > Z) { // mu > t
    double alpha = 0.0;
    while (r.unif() > alpha) {
      // X = t + 1.0;
      // while (X > t)
      // 	X = 1.0 / r.gamma_rate(0.5, 0.5);
      // Slightly faster to use truncated normal.
      double E1 = r.expon_rate(1.0); double E2 = r.expon_rate(1.0);
      while ( E1*E1 > 2 * E2 / t) {
	E1 = r.expon_rate(1.0); E2 = r.expon_rate(1.0);
      }
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = exp(-0.5 * Z*Z * X);
    }
  }
  else {
    double mu = 1.0 / Z;
    while (X > t) {
      double Y = r.norm(1.0); Y *= Y;
      double half_mu = 0.5 * mu;
      double mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if (r.unif() > mu / (mu + X))
	X = mu*mu / X;
    }
  }
  return X;
}

////////////////////////////////////////////////////////////////////////////////
				  // Sample //
////////////////////////////////////////////////////////////////////////////////

// double PolyaGammaPar::draw(double n, double z, RNG& r)
// {
//   return draw_sum_of_gammas(n, z, r);
// }

double PolyaGammaPar::draw(int n, double z, RNG& r)
{
  if (n < 1) throw std::invalid_argument("PolyaGammaPar::draw: n < 1.");
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += draw_like_devroye(z, r);
  return sum;
} // draw

double PolyaGammaPar::draw_sum_of_gammas(double n, double z, RNG& r)
{
  double x = 0;
  double kappa = z * z;
  for(int k=0; k < T; ++k)
    x += r.gamma_scale(n, 1.0) / (b[k] + kappa);
  return 2.0 * x;
} // draw_sum_of_gammas

double PolyaGammaPar::draw_like_devroye(double Z, RNG& r)
{
  // Change the parameter.
  Z = fabs(Z) * 0.5;

  // Now sample 0.25 * J^*(1, Z := Z/2).
  double fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  // ... Problems with large Z?  Try using q_over_p.
  // double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
  // double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

  double X = 0.0;
  double S = 1.0;
  double Y = 0.0;
  // int iter = 0; If you want to keep track of iterations.

  while (true) {

    // if (r.unif() < p/(p+q))
    if ( r.unif() < mass_texpon(Z) )
      X = __TRUNC + r.expon_rate(1) / fz;
    else
      X = rtigauss(Z, r);

    S = a(0, X);
    Y = r.unif() * S;
    int n = 0;
    bool go = true;

    // Cap the number of iterations?
    while (go) {

      // Break infinite loop.  Put first so it always checks n==0.
      #ifdef USE_R
      if (n % 1000 == 0) R_CheckUserInterrupt();
      #endif

      ++n;
      if (n%2==1) {
	S = S - a(n, X);
	if ( Y<=S ) return 0.25 * X;
      }
      else {
	S = S + a(n, X);
	if ( Y>S ) go = false;
      }

    }
    // Need Y <= S in event that Y = S, e.g. when X = 0.

  }
} // draw_like_devroye

void PolyaGammaPar::draw(double* samp, int* n, double* z, const int N, const int nthreads=1)
{
  #ifdef USE_R
  printf("You currently cannot use PolyaGammaPar without GSL.\n");
  return;
  #endif

  int i;
  
  #pragma omp parallel shared(samp, n, z) private(i) num_threads(nthreads)
  {
    // // Get thread number and write out.
    // tid = omp_get_thread_num();
    // printf("Hello from thread %i.\n", tid);
    // // Check if master thread.
    // if (tid==0) {
    //   nthreads = omp_get_num_threads();
    //   printf("There are %i threads.\n", nthreads);
    // }

    RNG r;

    // Now let's do this in parallel.
    #pragma omp for schedule(dynamic) nowait
    for (i = 0; i < N; i++) {
      samp[i] = draw(n[i], z[i], r);
    }

  }
}

////////////////////////////////////////////////////////////////////////////////
			       // END OF CLASS //
////////////////////////////////////////////////////////////////////////////////

#endif

////////////////////////////////////////////////////////////////////////////////
				 // APPENDIX //
////////////////////////////////////////////////////////////////////////////////

// It was not faster to use "vectorized" versions of r.gamma or r.igamma.
