#include "PolyaGammaSmallB.h"
#include <stdexcept>


////////////////////////////////////////////////////////////////////////////////
			       // Constructors //
////////////////////////////////////////////////////////////////////////////////

PolyaGammaSmallB::PolyaGammaSmallB() {}

////////////////////////////////////////////////////////////////////////////////
				 // Utility //
////////////////////////////////////////////////////////////////////////////////
inline double PolyaGammaSmallB::one_minus_psi(double x, double b)
{
    double omp = 1.0;
    omp -= (2.0+b) * exp(-2.*(b+1.0)/x);
    omp += (1.0+b)*(4.0+b)/2.0 * exp(-4.0*(b+2.0)/x);
    omp -= (2.0+b)*(1.0+b)*(6.0+b)/6.0 * exp(-6.0*(b+3.0)/x);
    omp += (3.0+b)*(2.0+b)*(1.0+b)*(8.0+b)/24.0 * exp(-8.0*(b+4.0)/x);
    omp -= (4.0+b)*(3.0+b)*(2.0+b)*(1.0+b)*(10.0+b)/120.0 * exp(-10.0*(b+5.0)/x);
    return omp;
}

////////////////////////////////////////////////////////////////////////////////
				  // Sample //
////////////////////////////////////////////////////////////////////////////////

double PolyaGammaSmallB::draw(double b, double z, RNG& r)
{
    double x;
    if (z == 0)
    {
        x = draw_invgamma_rej(b, r) / 4.0;
    }
    else
    {
        x = draw_invgauss_rej(b, z/2.0, r) / 4.0;
    }
    return x;
}

double PolyaGammaSmallB::draw_invgauss_rej(double b, double z, RNG& r)
{
    bool success = false;
    int niter = 0;

//    fprintf(stderr, "b: %.3f\t z: %.3f\n", b, z);
    double mu = b / fabs(z);
    double lambda = b * b;

    double x, u;

    while (!success && niter < _MAXITER)
    {
        x = r.igauss(mu, lambda);
        u = r.unif();
        if (u < one_minus_psi(x, b))
        {
            success = true;
        }
        niter += 1;
    }

    if (!success)
    {
        throw std::runtime_error("InvGauss rejection sampler failed for MAXITER iterations.");
    }

    return x;
}

double PolyaGammaSmallB::draw_invgamma_rej(double b, RNG& r)
{
    bool success = false;
    int niter = 0;

    double alpha = 0.5;
    double beta = b * b / 2.0;

    double x, u;

    while (!success && niter < _MAXITER)
    {
        x = r.igamma(alpha, beta);
        u = r.unif();
        if (u < one_minus_psi(x, b))
        {
            success = true;
        }
        niter += 1;
    }

    if (!success)
    {
        throw std::runtime_error("InvGamma rejection sampler failed for MAXITER iterations.");
    }

    return x;
}

