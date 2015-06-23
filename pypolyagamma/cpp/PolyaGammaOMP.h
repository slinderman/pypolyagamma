// -*- mode: c++; c-basic-offset: 4; -*-

#include "RNG.h"
#include "PolyaGammaHybrid.h"
#include <exception>
#include <stdio.h>
#include <omp.h>

#ifndef POLYAGAMMAOMP__
#define POLYAGAMMAOMP__

template <typename Real>
class PolyaGammaOMP
{

public:

    PolyaGammaOMP();
    PolyaGammaOMP(int nthread_);

    void draw_hybrid(Real* x, Real* b, Real* z, int num, vector<RNG>& r);
    void draw_devroye(Real* x, int* n, Real* z, int num, vector<RNG>& r);
    
private:

    vector< PolyaGammaHybrid<Real> > pg;

};

template <typename Real>
PolyaGammaOMP<Real>::PolyaGammaOMP()
    : pg(1)
{}

template <typename Real>
PolyaGammaOMP<Real>::PolyaGammaOMP(int nthread_)
    : pg(nthread_)
{}

template <typename Real>
void PolyaGammaOMP<Real>::draw_hybrid(Real *x, Real* b, Real* z, int num, vector<RNG>& r)
{
    #ifdef USE_R
    printf("Currently, you must use GSL for parallel draw.\n");
    return;
    #endif

    unsigned int nrng = r.size();

   // TODO: maybe need to set truncation for sum of gammas.
    if (nrng != pg.size()) {
	fprintf(stderr, "Warning: resizing PolyaGammaOMP to %i elements.\n", nrng);
	pg.resize(nrng);
    }

    int i, tid;

    vector<RNG>* rp = &r;
    vector< PolyaGammaHybrid<Real> > *pgp = &pg;

    #pragma omp parallel shared(x, b, z, rp, pgp) private(i, tid) num_threads(nrng)
    {
        // Get thread number and write out.
        tid = omp_get_thread_num();

	// fprintf(stderr, "Thread %i reporting.\n", tid);

        #pragma omp for schedule(dynamic) nowait
        for(int i=0; i < num; ++i){
	    x[i] = (*pgp)[tid].draw(b[i], z[i], (*rp)[tid]);
        }

    }

}

template <typename Real>
void PolyaGammaOMP<Real>::draw_devroye(Real* x, int* n, Real* z, int num, vector<RNG>& r)
{
    #ifdef USE_R
    printf("Currently, you must use GSL for parallel draw.\n");
    return;
    #endif

    int nrng = r.size();

    // TODO: maybe need to set truncation for sum of gammas.
    if (nrng != pg.size()) {
	fprintf(stderr, "Warning: resizing PolyaGammaOMP to %i elements.\n", nrng);
	pg.resize(nrng);
    }

    int i, tid;

    vector<RNG>* rp = &r;
    vector< PolyaGammaHybrid<Real> > *pgp = &pg;

    #pragma omp parallel shared(x, n, z, rp, pgp) private(i, tid) num_threads(nrng)
    {
        // // Get thread number and write out.
        tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic) nowait
        for(int i=0; i < num; ++i){
	    x[i] = (Real) (*pgp)[tid].dv.draw(n[i], (double) z[i], (*rp)[tid]);
        }

    }

}

#endif
