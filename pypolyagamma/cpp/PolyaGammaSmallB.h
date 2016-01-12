// -*- mode: c++; -*-

////////////////////////////////////////////////////////////////////////////////
// Scott Linderman
//
// Polya-gamma sampling in the small shape parameter regime.
////////////////////////////////////////////////////////////////////////////////

#ifndef __POLYAGAMMASMALLB__
#define __POLYAGAMMASMALLB__

#include "RNG.hpp"
#include <cmath>

#define _MAXITER 100

class PolyaGammaSmallB
{
    public:

    // Constructors.
    PolyaGammaSmallB();

    // Draw.
    double draw(double b, double z, RNG& r);

    private:

    double draw_invgauss_rej(double b, double z, RNG& r);
    double draw_invgamma_rej(double b, RNG& r);

    // Helper.
    inline double one_minus_psi(double x, double b);

};

#endif
