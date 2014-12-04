#include "PolyaGammaSP.h"
#include "InvertY.hpp"
#include <stdexcept>

//------------------------------------------------------------------------------

// double v_secant(double y, double vb, double va, double tol, int maxiter)
// {
//   double yb = y_func(vb);
//   double ya = y_func(va);

//   if (yb > ya) fprintf(stderr, "v_secant: yb(elow) > ya(above).\n");

//   int iter = 0;
//   double ydiff = tol + 1.;
//   double vstar, ystar;

//   while (fabs(ydiff) > tol && iter < maxiter) {
//     iter = iter + 1;
//     double m = (ya - yb) / (va - vb);
//     vstar = (y - yb) / m + vb;
//     ystar = y_func(vstar);
//     ydiff = y - ystar;
//     if (ystar < y) {
//       vb = vstar;
//       yb = ystar;
//     } else {
//       va = vstar;
//       ya = ystar;
//     }
//     // printf("y, v, ydiff: %g, %g, %g\n", ystar, vstar, ydiff);
//   }

//   if (iter >= maxiter) fprintf(stderr, "v_secant: reached maxiter.\n");

//   return vstar;
// }

// double v_func(double y) {
//   double lowerb = -100;
//   double upperb = 2.22;

//   double v = 0.0;
//   if (y < 0.1)
//     v = -1. / (y*y);
//   else if (y > 8.285225) {
//     v = atan(y * 0.5 * __PI);
//     v = v * v;
//   }
//   else
//     v = v_secant(y, lowerb, upperb, 1e-8, 10000);
//   return v;
// }

//------------------------------------------------------------------------------

double PolyaGammaSP::rtigauss(double mu, double lambda, double trunc, RNG& r)
{
  // mu = fabs(mu);
  double X = trunc + 1.0;
  if (trunc < mu) { // mu > t
    double alpha = 0.0;
    while (r.unif() > alpha) {
      X = r.rtinvchi2(lambda, trunc);
      alpha = exp(-0.5 * lambda / (mu*mu) * X);
    }
    // printf("rtigauss, part i: %g\n", X);
  }
  else {
    while (X > trunc) {
      X = r.igauss(mu, lambda);
    }
    // printf("rtigauss, part ii: %g\n", X);
  }
  return X;
}

double PolyaGammaSP::y_func(double v)
{
  double tol = 1e-6;
  double y   = 0.0;
  double r   = sqrt(fabs(v));
  if (v > tol)
    y = tan(r) / r;
  else if (v < -1*tol)
    y = tanh(r) / r;
  else
    y = 1 + (1/3) * v + (2/15) * v * v + (17/315) * v * v * v;
  return y;
}

double PolyaGammaSP::cos_rt(double v)
{
  double y   = 0.0;
  double r   = sqrt(fabs(v));
  if (v >= 0)
    y = cos(r);
  else
    y = cosh(r);
  return y;
}

void PolyaGammaSP::delta_func(double x, double mid, FD& delta)
{
  if (x >= mid) {
    delta.val = log(x) - log(mid);
    delta.der = 1.0 / x;
  }
  else {
    delta.val = 0.5 * (1 - 1.0 / x) - 0.5 * (1 - 1.0 / mid);
    delta.der = 0.5 / (x*x);
  }
}

double PolyaGammaSP::phi_func(double x, double z, FD& phi)
{
  // double v = yv.v_func(x);
  double v = v_eval(x);
  double u = 0.5 * v;
  double t = u + 0.5 * z*z;

  phi.val = log(cosh(fabs(z))) - log(cos_rt(v)) - t * x;
  phi.der = -1.0 * t;

  return v;
}

double PolyaGammaSP::tangent_to_eta(double x, double z, double mid, Line& tl)
{
  FD phi, delta, eta;
  double v;

  v = phi_func(x, z, phi);
  delta_func(x, mid, delta);

  eta.val = phi.val - delta.val;
  eta.der = phi.der - delta.der;

  // printf("v=%g\nphi=%g, phi.d=%g\ndelta=%g, delta.d=%g\neta=%g, eta.d=%g\n",
  // 	 v, phi.val, phi.der, delta.val, delta.der, eta.val, eta.der);

  tl.slope = eta.der;
  tl.icept = eta.val - eta.der * x;
  
  return v;
}

double PolyaGammaSP::sp_approx(double x, double n, double z)
{
  // double v  = yv.v_func(x);
  double v = v_eval(x);
  double u  = 0.5 * v;
  double z2 = z * z;
  double t  = u + 0.5 * z2;
  // double m  = y_func(-1 * z2);

  double phi = log(cosh(z)) - log(cos_rt(v)) - t * x;

  double K2  = 0.0;
  if (fabs(v) >= 1e-6) 
    K2 = x*x + (1-x) / v;
  else
    K2 = x*x - 1/3 - (2/15) * v;

  double log_spa = 0.5 * log(0.5 * n / __PI) - 0.5 * log(K2) + n * phi;
  return exp(log_spa);
}

int PolyaGammaSP::draw(double& d, double n, double z, RNG& r, int maxiter)
{
  if (n < 1) fprintf(stderr, "PolyaGammaSP::draw: n must be >= 1.\n");
  z = 0.5 * fabs(z);

  double xl = y_func(-1*z*z);    // Mode of phi - Left point.
  double md = xl * 1.1;          // Mid point.
  double xr = xl * 1.2;          // Right point.

  // printf("xl, md, xr: %g, %g, %g\n", xl, md, xr);

  // Inflation constants
  // double vmd  = yv.v_func(md);
  double vmd  = v_eval(md);
  double K2md = 0.0;

  if (fabs(vmd) >= 1e-6) 
    K2md = md*md + (1-md) / vmd;
  else
    K2md = md*md - 1/3 - (2/15) * vmd;
  
  double m2 = md * md;
  double al = m2*md / K2md;
  double ar = m2    / K2md;

  // printf("vmd, K2md, al, ar: %g, %g %g %g\n", vmd, K2md, al, ar);

  // Tangent lines info.
  Line ll, lr;
  tangent_to_eta(xl, z, md, ll);
  tangent_to_eta(xr, z, md, lr);

  double rl = -1. * ll.slope;
  double rr = -1. * lr.slope;
  double il = ll.icept;
  double ir = lr.icept;

  // printf("rl, rr, il, ir: %g, %g, %g, %g\n", rl, rr, il, ir);

  // Constants
  double lcn = 0.5 * log(0.5 * n / __PI);
  double rt2rl = sqrt(2 * rl);
  
  // printf("sqrt(rl): %g\n", rt2rl);

  // Weights
  double wl, wr, wt, pl;

  wl = exp(0.5 * log(al) - n * rt2rl + n * il + 0.5 * n * 1./md) * 
    RNG::p_igauss(md, 1./rt2rl, n);

  wr = exp(0.5 * log(ar) + lcn - n * log(n * rr) + n * ir - n * log(md)) *
    // yv.upperIncompleteGamma(md, n, n*rr);
    RNG::Gamma(n) * (1.0 - RNG::p_gamma_rate(md, n, n*rr));

  // printf("wl, wr: %g, %g\n", wl, wr);

  wt = wl + wr;
  pl = wl / wt;

  // Sample
  bool go  = true;
  int iter = 0;
  double X = 2.0;
  double F = 0.0;

  while(go && iter < maxiter) {
    // Put first so check on first pass.
    #ifdef USE_R
    if (iter % 1000 == 0) R_CheckUserInterrupt();
    #endif

    iter++;

    double phi_ev;
    if (r.unif() < pl) {
      X = rtigauss(1./rt2rl, n, md, r);
      phi_ev = n * (il - rl * X) + 0.5 * n * ((1.-1./X) - (1.-1./md));
      F = exp(0.5 * log(al) + lcn - 1.5 * log(X) + phi_ev);
    }
    else {
      X = r.ltgamma(n, n*rr, md);
      phi_ev = n * (ir - rr * X) + n * (log(X) - log(md));
      F = exp(0.5 * log(ar) + lcn + phi_ev) / X;
    }

    double spa = sp_approx(X, n, z);

    if (F * r.unif() < spa) go = false;

  }

  // return n * 0.25 * X;
  d = n * 0.25 * X;
  return iter;
}
