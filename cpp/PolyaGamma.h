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

#ifndef __POLYAGAMMA__
#define __POLYAGAMMA__

#include "RNG.hpp"
#include <cmath>
#include <vector>

using std::vector;

// The numerical accuracy of __PI will affect your distribution.
const double __PI = 3.141592653589793238462643383279502884197;
const double HALFPISQ = 0.5 * __PI * __PI;
const double FOURPISQ = 4 * __PI * __PI;
const double __TRUNC = 0.64;
const double __TRUNC_RECIP = 1.0 / __TRUNC;

class PolyaGamma
{

  // For sum of Gammas.
  int T;
  vector<double> bvec;

 public:

  // Constructors.
  PolyaGamma(int trunc = 200);

  // Draw.
  // double draw(double n, double z, RNG& r);
  double draw(int n, double z, RNG& r);
  double draw_sum_of_gammas(double n, double z, RNG& r);
  double draw_like_devroye(double z, RNG& r);

  //void draw(MF x, double a, double z, RNG& r);
  //void draw(MF x, MF a, MF z, RNG& r);

  // Utility.
  void set_trunc(int trunc);

  // Helper.
  double a(int n, double x);
  double pigauss(double x, double Z);
  double mass_texpon(double Z);
  double rtigauss(double Z, RNG& r);

  static double jj_m1(double b, double z);
  static double jj_m2(double b, double z);
  static double pg_m1(double b, double z);
  static double pg_m2(double b, double z);

};

#endif