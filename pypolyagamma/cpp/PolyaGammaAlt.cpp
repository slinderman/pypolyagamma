#include "PolyaGammaAlt.h"
#include <stdexcept>

//------------------------------------------------------------------------------

double rtinvchi2(double h, double trunc, RNG& r)
{
  double h2 = h * h;
  double R = trunc / h2;
  double X = 0.0;
  // I need to consider using a different truncated normal sampler.
  double E1 = r.expon_rate(1.0); double E2 = r.expon_rate(1.0);
  while ( (E1*E1) > (2 * E2 / R)) {
    // printf("E %g %g %g %g\n", E1, E2, E1*E1, 2*E2/R);
    E1 = r.expon_rate(1.0); E2 = r.expon_rate(1.0);
  }
  // printf("E %g %g \n", E1, E2);
  X = 1 + E1 * R;
  X = R / (X * X);
  X = h2 * X;
  return X;
}

//------------------------------------------------------------------------------

double PolyaGammaAlt::a_coef(int n, double x, double h)
{
  double d_n = 2.0 * (double) n + h;
  double log_out = h * log(2.0) - RNG::Gamma(h, true) + RNG::Gamma(n+h, true) \
    - RNG::Gamma(n+1, true) + log(d_n)					\
    - 0.5 * log(2.0 * __PI * x * x * x) - 0.5 * d_n * d_n / x;
  double out = exp(log_out);
  // double out = exp(out) is a legal command.  Weird.
  return out;
}

double PolyaGammaAlt::a_coef_recursive(double n, double x, double h, double coef_h, double& gnh_over_gn1_gh)
{
  double d_n = 2.0 * (double) n + h;
  // gamma_nh_over_n *= (n + h - 1) / n;  // Can speed up further by separate function for a0 and an, n > 0.
  if (n != 0)
    gnh_over_gn1_gh *= (n + h - 1) / n;
  else
    gnh_over_gn1_gh  = 1.0;
  double coef       = coef_h * gnh_over_gn1_gh;
  double log_kernel = - 0.5 * (log(x * x * x) + d_n * d_n / x) + log(d_n);
  return coef * exp(log_kernel);
  // double out = exp(out) is a legal command.  Weird.
}

double PolyaGammaAlt::pigauss(double x, double z, double lambda)
{
  // z = 1 / mean
  double b = sqrt(lambda / x) * (x * z - 1);
  double a = sqrt(lambda / x) * (x * z + 1) * -1.0;
  double y = RNG::p_norm(b) + exp(2 * lambda * z) * RNG::p_norm(a);
  return y;
}

double PolyaGammaAlt::w_left(double trunc, double h, double z)
{
  double out = 0;
  if (z != 0) 
    out = exp(h * (log(2.0) - z)) * pigauss(trunc, z/h, h*h);
  else
    out = exp(h * log(2.0)) * (1.0 - RNG::p_gamma_rate(1/trunc, 0.5, 0.5*h*h));
  return out;
}

double PolyaGammaAlt::w_right(double trunc, double h, double z)
{
  double lambda_z = PISQ * 0.125 + 0.5 * z * z;
  double p = exp(h * log(HALFPI / lambda_z)) * (1.0-RNG::p_gamma_rate(trunc, h, lambda_z));
  return p;
}

double PolyaGammaAlt::rtigauss(double h, double z, double trunc, RNG& r)
{
  z = fabs(z);
  double mu = h/z;
  double X = trunc + 1.0;
  if (mu > trunc) { // mu > t
    double alpha = 0.0;
    while (r.unif() > alpha) {
      X = rtinvchi2(h, trunc, r);
      alpha = exp(-0.5 * z*z * X);
    }
    // printf("rtigauss, part i: %g\n", X);
  }
  else {
    while (X > trunc) {
      X = r.igauss(mu, h*h);
    }
    // printf("rtigauss, part ii: %g\n", X);
  }
  return X;
}

double PolyaGammaAlt::g_tilde(double x, double h, double trunc)
{
  double out = 0;
  if (x > trunc) 
    out = exp(h * log(0.5 * __PI) + (h-1) * log(x) - PISQ * 0.125 * x - RNG::Gamma(h, true));
  else 
    out = h * exp( h * log(2.0) - 0.5 * log(2.0 * __PI * x * x * x) - 0.5 * h * h / x);
    // out = h * pow(2, h) * pow(2 * __PI * pow(x,3), -0.5) * exp(-0.5 * pow(h,2) / x);
  return out;
}

////////////////////////////////////////////////////////////////////////////////
				  // Sample //
////////////////////////////////////////////////////////////////////////////////

double PolyaGammaAlt::draw_abridged(double h, double z, RNG& r, int max_inner)
{
  if (h < 1 || h > 4) {
    fprintf(stderr, "PolyaGammaAlt::draw h = %g must be in [1,4]\n", h);
    return 0;
  }

  // Change the parameter.
  z = fabs(z) * 0.5;
  
  int    idx   = (int) floor((h-1.0)*100.0);
  double trunc = trunc_schedule[idx];

  // Now sample 0.25 * J^*(1, z := z/2).
  double rate_z       = 0.125 * __PI*__PI + 0.5 * z*z;
  double weight_left  = w_left (trunc, h, z);
  double weight_right = w_right(trunc, h, z);
  double prob_right   = weight_right / (weight_right + weight_left);

  // printf("prob_right: %g\n", prob_right);
  
  double coef1_h = exp(h * log(2.0) - 0.5 * log(2.0 * __PI));
  // double gamma_nh_over_n = RNG::Gamma(h);
  double gnh_over_gn1_gh = 1.0; // Will fill in value on first call to a_coef_recursive.

  int num_trials = 0;
  int total_iter = 0;

  while (num_trials < 10000) {
    num_trials++;

    double X = 0.0;
    double Y = 0.0;

    // if (r.unif() < p/(p+q))
    double uu = r.unif();
    if ( uu < prob_right )
      X = r.ltgamma(h, rate_z, trunc);
    else
      X = rtigauss(h, z, trunc, r);

    // double S  = a_coef(0, X, h);
    double S = a_coef_recursive(0.0, X, h, coef1_h, gnh_over_gn1_gh);
    double a_n = S;
    // double a_n2 = S2;
    // printf("a_n=%g, a_n2=%g\n", a_n, a_n2);
    double gt =  g_tilde(X, h, trunc);
    Y = r.unif() * gt;

    // printf("test gt: %g\n", g_tilde(trunc * 0.1, h, trunc));
    // printf("X, Y, S, gt: %g, %g, %g, %g\n", X, Y, S, gt);

    bool decreasing = false;

    int  n  = 0;
    bool go = true;

    // Cap the number of iterations?
    while (go && n < max_inner) {
      total_iter++;

      // Break infinite loop.  Put first so it always checks n==0.
      #ifdef USE_R
      if (n % 1000 == 0) R_CheckUserInterrupt();
      #endif

      ++n;
      double prev = a_n;
      // a_n  = a_coef(n, X, h);
      a_n = a_coef_recursive((double)n, X, h, coef1_h, gnh_over_gn1_gh);
      // printf("a_n=%g, a_n2=%g\n", a_n, a_n2);
      decreasing = a_n <= prev;

      if (n%2==1) {
	S = S - a_n;
	if ( Y<=S && decreasing) return 0.25 * X;
      }
      else {
	S = S + a_n;
	if ( Y>S && decreasing) go = false;
      }

    }
    // Need Y <= S in event that Y = S, e.g. when X = 0.

  }
  
  // We should never get here.
  return -1.0;
} // draw

double PolyaGammaAlt::draw(double h, double z, RNG& r, int max_inner)
{
  if (h < 1) {
    fprintf(stderr, "PolyaGammaAlt::draw h = %g must be >= 1\n", h);
    return 0;
  }

  double n = floor( (h-1.0) / 4.0 );
  double remain = h - 4.0 * n;

  double x = 0.0;

  for (int i = 0; i < (int)n; i++) 
    x += draw_abridged(4.0, z, r);
  if (remain > 4.0)
    x += draw_abridged(0.5 * remain, z, r) + draw_abridged(0.5 * remain, z, r);
  else
    x += draw_abridged(remain, z, r);

  return x;
}

////////////////////////////////////////////////////////////////////////////////
				 // APPENDIX //
////////////////////////////////////////////////////////////////////////////////

// We should only have to calculate Gamma(h) once.  We can then get Gamma(n+h)
// from the recursion Gamma(z+1) = z Gamma(z).  Not sure how that is in terms of
// stability, but that should save us quite a few computations.  This affects
// a_coef and g_tilde.
