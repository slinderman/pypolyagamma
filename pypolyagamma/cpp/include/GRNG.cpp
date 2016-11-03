#include "GRNG.hpp"

//////////////////////////////////////////////////////////////////////
			  // Constructors //
//////////////////////////////////////////////////////////////////////

BasicRNG::BasicRNG(unsigned long seed)
{
  r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set (r, seed);
}


//////////////////////////////////////////////////////////////////////
			  // Assignment= //
//////////////////////////////////////////////////////////////////////

BasicRNG& BasicRNG::operator=(const BasicRNG& rng)
{
  // The random number generators must be of the same type.
  gsl_rng_memcpy(r, rng.r );
  return *this;
}

//////////////////////////////////////////////////////////////////////
			  // Read / Write //
//////////////////////////////////////////////////////////////////////


bool BasicRNG::read(const string& filename)
{
    return true;
} // Read

bool BasicRNG::write(const string& filename){
    return true;
} // Write

void BasicRNG::set(unsigned long seed)
{
    gsl_rng_set(r, seed);
} // Set

//////////////////////////////////////////////////////////////////////
		      // GSL Random Variates //
//////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------
// Distributions with one parameter.

#define ONEP(NAME, CALL, P1)			\
  double BasicRNG::NAME(double P1)	\
  {						\
    return CALL (r, P1);			\
  }						\

ONEP(expon_mean, gsl_ran_exponential, mean)
ONEP(chisq,  gsl_ran_chisq      , df  )
ONEP(norm,   gsl_ran_gaussian   , sd  )

#undef ONEP

//--------------------------------------------------------------------
// Distributions with two parameters.

#define TWOP(NAME, CALL, P1, P2)			\
  double BasicRNG::NAME(double P1, double P2)	\
  {							\
    return CALL (r, P1, P2);				\
  }							\

TWOP(gamma_scale, gsl_ran_gamma_knuth, shape, scale)
TWOP(flat , gsl_ran_flat , a    , b    )
TWOP(beta , gsl_ran_beta , a    , b    )

// x ~ Gamma(shape=a, scale=b)
// x ~ x^{a-1} exp(x / b).

#undef TWOP

//////////////////////////////////////////////////////////////////////
		     // Custom Random Variates //
//////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------
			   // Bernoulli //

int BasicRNG::bern(double p)
{
  return gsl_ran_bernoulli(r, p);
}

//--------------------------------------------------------------------
			    // Uniform //

double BasicRNG::unif()
{
  return gsl_rng_uniform(r);
} // unif

//--------------------------------------------------------------------
			  // Exponential //
double BasicRNG::expon_rate(double rate)
{
  return expon_mean(1.0 / rate);
}

//--------------------------------------------------------------------
			    // Normal //

double BasicRNG::norm(double mean, double sd)
{
  return mean + gsl_ran_gaussian(r, sd);
} // norm

//--------------------------------------------------------------------
			   // Gamma_Rate //

double BasicRNG::gamma_rate(double shape, double rate)
{
  return gamma_scale(shape, 1.0 / rate);
}

//--------------------------------------------------------------------
			   // Inv-Gamma //

// a = shape, b = scale
// x ~ IG(shape, scale) ~ x^{-a-1} exp(b / x).
// => 1/x ~ Ga(shape, 1/scale).

double BasicRNG::igamma(double shape, double scale)
{
  return 1.0/gsl_ran_gamma_knuth(r, shape, 1.0/scale);
} // igamma

////////////////////////////////////////////////////////////////////////////////

double BasicRNG::p_norm(double x, int use_log)
{
  double m = gsl_cdf_ugaussian_P(x);
  if (use_log) m = log(m);
  return m;
}

double BasicRNG::p_gamma_rate(double x, double shape, double rate, int use_log)
{
  double scale = 1.0 / rate;
  double y = gsl_cdf_gamma_P(x, shape, scale);
  if (use_log) y = log(y);
  return y;
}

////////////////////////////////////////////////////////////////////////////////

double BasicRNG::Gamma(double x, int use_log)
{
  double y = gsl_sf_lngamma(x);
  if (!use_log) y = exp(y);
  return y;
}

////////////////////////////////////////////////////////////////////////////////

double BasicRNG::d_beta(double x, double a, double b)
{
  return gsl_ran_beta_pdf(x, a, b);
}

////////////////////////////////////////////////////////////////////////////////
