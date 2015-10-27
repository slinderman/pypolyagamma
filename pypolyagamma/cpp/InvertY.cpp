#include "InvertY.hpp"
#include <stdio.h>

#ifdef USE_R
#include "R.h"
#endif

//------------------------------------------------------------------------------

double y_eval(double v)
{
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

void ydy_eval(double v, double* yp, double* dyp)
{
  // double r   = sqrt(fabs(v));

  double y = y_eval(v);
  *yp = y;

  if (fabs(v) >= tol)
    *dyp = 0.5 * (y*y + (1-y) / v);
  else
    *dyp = 0.5 * (y*y - 1/3 - (2/15) * v);

}

double f_eval(double v, void * params)
{
  double y = *((double*) params);
  return y_eval(v) - y;
}

void fdf_eval(double v, void* params, double* fp, double* dfp)
{
  double y = *((double*)params);
  ydy_eval(v, fp, dfp);
  *fp  -= y;
}

double df_eval(double v, void * params)
{
  double f, df;
  ydy_eval(v, &f, &df);
  return df;
}

double v_eval(double y, double tol, int max_iter)
{
  double ylower = ygrid[0];
  double yupper = ygrid[grid_size-1];

  if (y < ylower) {
    return -1. / (y*y);
  } else if (y > yupper) {
    double v = atan(0.5 * y * IYPI);
    return v*v;
  }
  else if (y==1) return 0.0;

  double id = (log(y) / log(2.0) + 4.0) / 0.1;
  // printf("y, id, y[id], v[id]: %g, %g, %g, %g\n", y, id, ygrid[(int)id], vgrid[(int)id]);

  // C++ default is truncate decimal portion.
  int idlow  = (int)id;
  int idhigh = (int)id + 1;
  double vl  = vgrid[idlow];  // lower bound
  double vh  = vgrid[idhigh]; // upper bound

  int    iter = 0;
  double diff = tol + 1.0;
  double vnew = vl;
  double vold = vl;
  double f0, f1;

  while (diff > tol && iter < max_iter) {
    iter++;
    vold = vnew;
    fdf_eval(vold, &y, &f0, &f1);
    vnew = vold - f0 / f1;
    vnew = vnew > vh ? vh : vnew;
    vnew = vnew < vl ? vl : vnew;
    diff = fabs(vnew - vold);
    // printf("iter: %i, v: %g, diff: %g\n", iter, vnew, diff);
  }

  if (iter >= max_iter) fprintf(stderr, "InvertY.cpp, v_eval: reached max_iter: %i\n", iter);

  return vnew;
}