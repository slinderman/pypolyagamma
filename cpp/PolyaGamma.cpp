#include "PolyaGamma.h"
#include <stdexcept>

using std::pow;

////////////////////////////////////////////////////////////////////////////////
				 // Utility //
////////////////////////////////////////////////////////////////////////////////


double a(int n, double x)
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

double pigauss(double x, double Z)
{
  double b = sqrt(1.0 / x) * (x * Z - 1);
  double a = sqrt(1.0 / x) * (x * Z + 1) * -1.0;
  double y = RNG::p_norm(b) + exp(2 * Z) * RNG::p_norm(a);
  return y;
}

double mass_texpon(double Z)
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

double rtigauss(double Z, RNG* r)
{
  Z = fabs(Z);
  double t = __TRUNC;
  double X = t + 1.0;
  if (__TRUNC_RECIP > Z) { // mu > t
    double alpha = 0.0;
    while (r->unif() > alpha) {
      // X = t + 1.0;
      // while (X > t)
      // 	X = 1.0 / r->gamma_rate(0.5, 0.5);
      // Slightly faster to use truncated normal.
      double E1 = r->expon_rate(1.0); double E2 = r->expon_rate(1.0);
      while ( E1*E1 > 2 * E2 / t) {
	E1 = r->expon_rate(1.0); E2 = r->expon_rate(1.0);
      }
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = exp(-0.5 * Z*Z * X);
    }
  }
  else {
    double mu = 1.0 / Z;
    while (X > t) {
      double Y = r->norm(1.0); Y *= Y;
      double half_mu = 0.5 * mu;
      double mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if (r->unif() > mu / (mu + X))
	X = mu*mu / X;
    }
  }
  return X;
}

////////////////////////////////////////////////////////////////////////////////
				  // Sample //
////////////////////////////////////////////////////////////////////////////////

double draw(int n, double z, RNG* r)
{
  if (n < 1) {
  #ifndef NTHROW
    throw std::invalid_argument("draw: n < 1.");
  #else
    fprintf(stderr, "draw: n < 1.  Set n = 1.\n");
    n = 1;
  #endif
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += draw_like_devroye(z, r);
  return sum;
} // draw

double draw_like_devroye(double Z, RNG* r)
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
    if ( r->unif() < mass_texpon(Z) )
      X = __TRUNC + r->expon_rate(1) / fz;
    else
      X = rtigauss(Z, r);

    S = a(0, X);
    Y = r->unif() * S;
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

////////////////////////////////////////////////////////////////////////////////
			      // Static Members //
////////////////////////////////////////////////////////////////////////////////

double jj_m1(double b, double z)
{
    z = fabs(z);
    double m1 = 0.0;
    if (z > 1e-12)
	m1 = b * tanh(z) / z;
    else
	m1 = b * (1 - (1.0/3) * pow(z,2) + (2.0/15) * pow(z,4) - (17.0/315) * pow(z,6));
    return m1;
}

double jj_m2(double b, double z)
{
    z = fabs(z);
    double m2 = 0.0;
    if (z > 1e-12)
	m2 = (b+1) * b * pow(tanh(z)/z,2) + b * ((tanh(z)-z)/pow(z,3));
    else
	m2 = (b+1) * b * pow(1 - (1.0/3) * pow(z,2) + (2.0/15) * pow(z,4) - (17.0/315) * pow(z,6), 2) +
	    b * ((-1.0/3) + (2.0/15) * pow(z,2) - (17.0/315) * pow(z,4));
    return m2;
}

double pg_m1(double b, double z)
{
    return jj_m1(b, 0.5 * z) * 0.25;
}
 
double pg_m2(double b, double z)
{
    return jj_m2(b, 0.5 * z) * 0.0625;
}
