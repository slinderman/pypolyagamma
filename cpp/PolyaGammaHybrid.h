// -*- mode: c++; c-basic-offset: 4; -*-

#include "RNG.hpp"
#include "PolyaGamma.h"
#include "PolyaGammaAlt.h"
#include "PolyaGammaSP.h"

#ifndef POLYAGAMMAHYBRID__
#define POLYAGAMMAHYBRID__

template <typename Real>
class PolyaGammaHybrid
{

public:

    PolyaGamma    dv;
    PolyaGammaAlt al;
    PolyaGammaSP  sp;

    Real draw(Real b, Real z, RNG& r);

};

template <typename Real>
Real PolyaGammaHybrid<Real>::draw(Real b_, Real z_, RNG& r)
{
    double x;

    double b = (double) b_;
    double z = (double) z_;

    if (b > 170) {
	double m = dv.pg_m1(b,z);
	double v = dv.pg_m2(b,z) - m*m;
	x = (Real) r.norm(m, sqrt(v));
    }
    else if (b > 13) {
	sp.draw(x, b, z, r);
    }
    else if (b==1 || b==2) {
	x = dv.draw((int)b, z, r);
    }
    else if (b > 1) {
	x = al.draw(b, z, r);
    }
    else if (b > 0) {
	x = dv.draw_sum_of_gammas(b, z, r);
    }
    else {
	x = 0.0;
    }

    return (Real) x;
}

typedef PolyaGammaHybrid<float>  PolyaGammaHybridFloat;
typedef PolyaGammaHybrid<double> PolyaGammaHybridDouble;

#endif
