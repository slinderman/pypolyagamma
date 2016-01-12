// -*- mode: c++; c-basic-offset: 4; -*-

#include "RNG.hpp"
#include "PolyaGamma.h"
#include "PolyaGammaAlt.h"
#include "PolyaGammaSP.h"
#include "PolyaGammaSmallB.h"

#ifndef POLYAGAMMAHYBRID__
#define POLYAGAMMAHYBRID__

template <typename Real>
class PolyaGammaHybrid
{
private:
    RNG*          rng;

public:
    // Constructor and destructor
    PolyaGammaHybrid(unsigned long seed);
    ~PolyaGammaHybrid();


    PolyaGamma       dv;
    PolyaGammaAlt    al;
    PolyaGammaSP     sp;
    PolyaGammaSmallB sb;

    void set_trunc(int trunc);
    Real draw(Real b, Real z);

};

// Constructor and Destructor
template <typename Real>
PolyaGammaHybrid<Real>::PolyaGammaHybrid(unsigned long seed)
{
    rng = new RNG(seed);
}

template <typename Real>
PolyaGammaHybrid<Real>::~PolyaGammaHybrid()
{
    delete rng;
}

// Plumbing
template <typename Real>
void PolyaGammaHybrid<Real>::set_trunc(int trunc)
{
    dv.set_trunc(trunc);
}

// Draw
template <typename Real>
Real PolyaGammaHybrid<Real>::draw(Real b_, Real z_)
{
    double x;

    double b = (double) b_;
    double z = (double) z_;

    if (b > 170)
    {
        double m = dv.pg_m1(b,z);
        double v = dv.pg_m2(b,z) - m*m;
        x = (Real) rng->norm(m, sqrt(v));
    }
    else if (b > 13)
    {
	    sp.draw(x, b, z, *rng);
    }
    else if (b==1 || b==2)
    {
	    x = dv.draw((int)b, z, *rng);
    }
    else if (b > 1)
    {
	    x = al.draw(b, z, *rng);
    }
    else if (b > 0)
    {
//        x = dv.draw_sum_of_gammas(b, z, *rng);
	    x = sb.draw(b, z, *rng);
    }
    else
    {
	    x = 0.0;
    }

    return (Real) x;
}

typedef PolyaGammaHybrid<float>  PolyaGammaHybridFloat;
typedef PolyaGammaHybrid<double> PolyaGammaHybridDouble;

#endif
