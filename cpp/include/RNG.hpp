// -*- c-basic-offset: 4; -*-
// Copyright 2013 Jesse Windle - jesse.windle@gmail.com

// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of
// the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see
// <http://www.gnu.org/licenses/>.

//////////////////////////////////////////////////////////////////////

#ifndef __RNG__
#define __RNG__

#include <stdio.h>
#include <stdexcept>
#include <cmath>

#ifdef USE_R
#include "RRNG.hpp"
#define  RCHECK 1000
#elif  USE_GRNGPAR
#include "GRNGPar.hpp"
#else
#include "GRNG.hpp"
#endif

// #ifndef __MYMAT__
// #define __MYMAT__
// typedef MatrixFrame MyMat;
// #endif

const double SQRT2PI = 2.50662827;

inline void check_R_interupt(int& count);

class RNG : public BasicRNG {

protected:

  // Truncated Normal Helper Functions / Variables.
  double alphastar(double left);
  double lowerbound(double left);

  // Truncated Right Gamma Helper Functions.
  double omega_k(int k, double a, double b);

 public:

 RNG(unsigned long seed);

  // Random variates.  I need to do this so I can overload the function names.
  using BasicRNG::unif;
  using BasicRNG::expon_mean;
  using BasicRNG::expon_rate;
  using BasicRNG::chisq ;
  using BasicRNG::norm  ;
  using BasicRNG::gamma_scale ;
  using BasicRNG::gamma_rate  ;
  using BasicRNG::igamma;
  using BasicRNG::flat  ;
  using BasicRNG::beta  ;

  using BasicRNG::p_norm;
  using BasicRNG::p_gamma_rate;
  
  using BasicRNG::Gamma;

  static double p_igauss(double x, double mu, double lambda);

  static double Beta(double a, double b, bool log=false);

  // Truncated Exponential
  double texpon_rate(double left, double rate);
  double texpon_rate(double left, double right, double rate);

  // Truncated Normal
  double tnorm(double left);               // One sided standard.
  double tnorm(double left, double right); // Two sided standard.
  double tnorm(double left, double mu, double sd);
  double tnorm(double left, double right, double mu, double sd);

  // Right tail of normal
  double tnorm_tail(double t);

  // Right truncated gamma.
  double right_tgamma_reject(double shape, double rate);
  double right_tgamma_beta(double shape, double rate);
  double rtgamma_rate(double shape, double rate, double right);

  // Left truncated gamma.
  double ltgamma(double shape, double rate, double trunc);

  // Inverse Gaussian.
  double igauss(double mu, double lambda);
  
  // Truncated Inv chi 2
  double rtinvchi2(double scale, double trunc);

  // Random variates with Mat.  Fills the Mat with samples.  Need to keep for legacy code.
  template<typename Mat> void unif  (Mat& M);
  template<typename Mat> void expon_mean(Mat& M, double mean);
  template<typename Mat> void expon_rate (Mat& M, double rate);
  template<typename Mat> void chisq (Mat& M, double df);
  template<typename Mat> void norm  (Mat& M, double sd);
  template<typename Mat> void norm  (Mat& M, double mean , double sd);
  template<typename Mat> void gamma_scale (Mat& M, double shape, double scale);
  template<typename Mat> void gamma_rate  (Mat& M, double shape, double scale);
  template<typename Mat> void igamma(Mat& M, double shape, double scale);
  template<typename Mat> void flat  (Mat& M, double a    , double b    );
  template<typename Mat> void tnorm (Mat& M, double left, double mu, double sd);
  template<typename Mat> void tnorm (Mat& M, double left, double right, double mu, double sd);

  template<typename Mat> void expon_mean(Mat& M, const Mat& mean);
  template<typename Mat> void expon_rate (Mat& M, const Mat& rate);
  template<typename Mat> void chisq (Mat& M, const Mat& df);
  template<typename Mat> void norm  (Mat& M, const Mat& sd);
  template<typename Mat> void norm  (Mat& M, const Mat& mean , const Mat& sd);
  template<typename Mat> void gamma_scale (Mat& M, const Mat& shape, const Mat& scale);
  template<typename Mat> void gamma_rate  (Mat& M, const Mat& shape, const Mat& scale);
  template<typename Mat> void igamma(Mat& M, const Mat& shape, const Mat& scale);
  template<typename Mat> void flat  (Mat& M, const Mat& a    , const Mat& b);

}; // RNG

////////////////////////////////////////////////////////////////////////////////
			   // BASIC RANDOM VARIATE //
////////////////////////////////////////////////////////////////////////////////

#ifndef uint
typedef unsigned int uint;
#endif

template<typename Mat>
void RNG::unif(Mat& M)
{
  for(uint i = 0; i < M.size(); ++i)
    M(i) = BasicRNG::flat();
} // unif

#define ONEP(FUNC, P1)				\
  template<typename Mat>			\
  void RNG::FUNC(Mat& M, double P1)		\
  {						\
    for(uint i = 0; i < (uint)M.size(); i++)	\
      M(i) = FUNC (P1);				\
  }						\
  template<typename Mat>			\
  void RNG::FUNC(Mat& M, const Mat& P1)		\
  {						\
    for(uint i = 0; i < (uint)M.size(); i++)	\
      M(i) = FUNC (P1(i % P1.size()));		\
  }						\

ONEP(expon_mean, mean)
ONEP(expon_rate, rate)
ONEP(chisq     ,   df)
ONEP(norm      ,   sd)

#undef ONEP

//--------------------------------------------------------------------
// Distributions with two parameters.

#define TWOP(FUNC, P1, P2)					\
  template<typename Mat>					\
  void RNG::FUNC(Mat& M, double P1, double P2)			\
  {								\
    for(uint i = 0; i < (uint)M.size(); i++)			\
      M(i) = FUNC (P1, P2);					\
  }								\
  template<typename Mat>					\
  void RNG::FUNC(Mat& M, const Mat& P1, const Mat& P2)		\
  {								\
    uint p1len = P1.size();					\
    uint p2len = P2.size();					\
    for(uint i = 0; i < (uint)M.size(); i++)			\
      M(i) = FUNC (P1(i%p1len), P2(i%p2len) );			\
  }								\

TWOP(norm       ,  mean,  sd)
TWOP(gamma_scale, shape,  scale)
TWOP(gamma_rate , shape,  scale)
TWOP(igamma     , shape,  scale)
TWOP(flat       ,     a,  b    )

#undef TWOP

template<typename Mat> void RNG::tnorm (Mat& M, double left, double mu, double sd){
  for(uint i = 0; i < (uint)M.size(); i++)	
    M(i) = tnorm(left, mu, sd);
}

template<typename Mat> void RNG::tnorm (Mat& M, double left, double right, double mu, double sd)
{
  for(uint i = 0; i < (uint)M.size(); i++) 
    M(i) = tnorm(left, right, mu, sd);
}

#endif
