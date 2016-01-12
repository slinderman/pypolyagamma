
/*********************************************************************
  Scott Linderman, 2015
  This class wraps c++11's random number generator and random
  distribution functions into a class.
  When compiling include -lgsl -lcblas -llapack .
*********************************************************************/

#ifndef __SRNG__
#define __SRNG__

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <cmath>
#include <random>

using std::string;
using std::ofstream;
using std::ifstream;

//////////////////////////////////////////////////////////////////////
			      // RNG //
//////////////////////////////////////////////////////////////////////

class BasicRNG {

 protected:

  gsl_rng * r;

 public:

  // Constructors and destructors.
  //BasicRNG();
  BasicRNG(unsigned long seed);
  //BasicRNG(const BasicRNG& rng);

  virtual ~BasicRNG()
    { gsl_rng_free (r); }

  // Assignment=
  BasicRNG& operator=(const BasicRNG& rng);

  // Read / Write / Set
  bool read (const string& filename);
  bool write(const string& filename);
  void set(unsigned long seed);

  // Get rng -- be careful.  Needed for other random variates.
  gsl_rng* getrng() { return r; }

  // Random variates.
  double unif  ();                             // Uniform
  double expon_mean(double mean);     // Exponential
  double expon_rate(double rate);                  // Exponential
  double chisq (double df);                    // Chisq
  double norm  (double sd);                    // Normal
  double norm  (double mean , double sd);      // Normal
  double gamma_scale (double shape, double scale); // Gamma_Scale
  double gamma_rate  (double shape, double rate);  // Gamma_Rate
  double igamma(double shape, double scale);   // Inv-Gamma
  double flat  (double a=0  , double b=1  );   // Flat
  double beta  (double a=1.0, double b=1.0);   // Beta

  int bern  (double p);                     // Bernoulli

  // CDF
  static double p_norm (double x, int use_log=0);
  static double p_gamma_rate(double x, double shape, double rate, int use_log=0);

  // Density
  static double d_beta(double x, double a, double b);

  // Utility
  static double Gamma (double x, int use_log=0);

}; // BasicRNG

#endif

////////////////////////////////////////////////////////////////////////////////
				 // APPENDIX //
////////////////////////////////////////////////////////////////////////////////

// If you make everything inline within the same translation unit then that
// function will not be callable from anohter translation unit.  You can see
// that the function is missing by using the nm command.
