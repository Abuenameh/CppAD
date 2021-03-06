/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 15 August 2015, 18:16
 */

#define CPPAD_CG_SYSTEM_LINUX 1

#include <cstdlib>
#include <vector>
#include <set>

using namespace std;

#include <boost/date_time.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
using namespace boost::posix_time;
using boost::thread_group;
using boost::lexical_cast;

//#include <cppad/cg/cppadcg.hpp>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <cppad/local/ad_fun.hpp>
//#include <cppad/ipopt/solve_callback.hpp>
//using namespace CppAD;
using CppAD::AD;

typedef CppAD::AD<double> Scalar;

#include <nlopt.hpp>

#include <lbfgs.h>

const int L = 50;
const int nmax = 5;
const int dim = nmax + 1;

template<class T>
complex<T> operator~(const complex<T> a) {
    return conj(a);
}

namespace CppAD {

    inline bool isinf(CppAD::AD<double> x) {
        return false;
    }

    inline double copysign(CppAD::AD<double> x, CppAD::AD<double> y) {
        return 0;
    }
}

template<class _Tp>
inline
complex<CppAD::AD<_Tp>>
operator*(const complex<CppAD::AD<_Tp>>&__x, const _Tp& __y) {
    complex<CppAD::AD < _Tp >> __t(__x);
    __t *= CppAD::AD<_Tp>(__y);
    return __t;
}

template<class _Tp>
inline
complex<CppAD::AD<_Tp>>
operator*(const _Tp& __x, const complex<CppAD::AD<_Tp>>&__y) {
    complex<CppAD::AD < _Tp >> __t(__y);
    __t *= CppAD::AD<_Tp>(__x);
    return __t;
}

inline int mod(int i) {
    return (i + L) % L;
}

inline double g2(int n, int m) {
    return sqrt(1.0 * (n + 1) * m);
}

inline double eps(vector<double>& U, int i, int j, int n, int m) {
    return n * U[i] - (m - 1) * U[j];
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}

inline double g(int n, int m) {
    return sqrt(1.0 * (n + 1) * m);
}

//inline Scalar eps(vector<double>& U, int i, int j, int n, int m) {
//	return n * U[i] - (m - 1) * U[j];
//}
//
//inline Scalar eps(double U, int n, int m) {
//    return (n - m + 1) * U;
//}

inline double eps(vector<double>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n * U[i] - (m - 1) * U[j] + (q - 1) * U[k] - p * U[l];
}

AD<double> norm(int i, vector<AD<double>>&fin) {

    complex<AD<double>>*f;
    AD<double> norm2 = 0;
    f = reinterpret_cast<complex<AD<double>>*> (&fin[2 * i * dim]);
    for (int n = 0; n <= nmax; n++) {
        norm2 += norm(f[n]); //f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
    }
    return norm2;
}

double norm(int i, vector<double>& fin) {

    complex<double>* f;
    double norm2 = 0;
    f = reinterpret_cast<complex<double>*> (&fin[2 * i * dim]);
    for (int n = 0; n <= nmax; n++) {
        norm2 += norm(f[n]); //f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
    }
    return norm2;
}

double abs(int i, vector<double>& fin) {

    complex<double>* f;
    double norm2 = 0;
    f = reinterpret_cast<complex<double>*> (&fin[2 * i * dim]);
    for (int n = 0; n <= nmax; n++) {
        norm2 += norm(f[n]); //f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
    }
    return sqrt(norm2);
}

template<class T>
T energy(CppAD::vector<T>& fin, double theta) {//, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {

    double U0 = 1;

    vector<double> dU({0.043199239962729674,0.08259186491719439,0.19397213987526984,0.3141217180800817,
    0.09151898766150635,0.32699155669771907,0.039702210154997974,0.3169288596880999,
    -0.06649698521376712,0.11030126264837037,0.13055555283210518,-0.10016340121284928,
    -0.05446325862556434,-0.17320924136983562,0.3589154219523414,-0.364378648638469,
    0.42719117976471876,-0.19375941242682493,-0.10096466795211789,-0.01959237427779159,
    0.2649228285617291,0.2840213898637185,0.02551000144713278,-0.017569415902371932,
    0.060356175879903606,-0.09299207573504409,0.39096937992083625,0.30643104688443557,
    -0.35352081276155445,-0.13992932138795577,-0.34123939491588917,0.1326266895645576,
    -0.39172920394431665,-0.113887642455067,0.3031844422912884,0.42098839070456484,
    0.25228016569989764,-0.3000213515178055,0.33836311285218534,0.33843394927349,0.4414618768627816,
    -0.023139212960446076,0.27186045802215664,0.27349689504696806,-0.03372008677117977,
    0.018059966538258365,0.2544888120684825,0.16063922586350765,-0.31722957528124107,
    0.1989880214046187});
    vector<double> J({0.019987202468844147,0.021496262642595524,0.023880344965865895,0.02275043182020429,
    0.022872374734557444,0.022282798105247224,0.022189944005260767,0.020950153751591195,
    0.019094560020353667,0.021156522094292012,0.018909357581322637,0.01718303541262763,
    0.016430926968766763,0.01998802665816625,0.017417518425073424,0.01789531552756776,
    0.02026580694225929,0.015786032522964198,0.017509454115343995,0.021035343501826984,
    0.02433180840900716,0.021724104274918832,0.0188015915058845,0.019141358143416474,
    0.018345120914591926,0.021262600165423456,0.025877859315358118,0.017195760116303044,
    0.013736970054750698,0.013872262466195355,0.01606824963829334,0.01541550459088139,
    0.013520421904084281,0.020262193793735988,0.026149396921297762,0.025586177019853436,
    0.017513826385303643,0.01816279319888601,0.025672873363203112,0.026746152903493584,
    0.022558932355961205,0.02105752138357231,0.02429506066325684,0.020950349337463866,
    0.018563602345284414,0.021366292117123837,0.022927132213120833,0.016588609400903988,
    0.016883447517063933,0.021120753167471747});
   for (int i = 0; i < L; i++) {
//       J[i] *= 5;
   }
        double mu = 0.9;

    complex<T> expth = complex<T>(cos(theta), sin(theta));
    complex<T> expmth = ~expth;
    complex<T> exp2th = expth*expth;
    complex<T> expm2th = ~exp2th;

    vector<complex<T>* > f(L);
    vector<T> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        f[j] = reinterpret_cast<complex<T>*> (&fin[2 * j * dim]);
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += norm(f[j][m]); //f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<T> E = complex<T>(0, 0);

    complex<T> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        Ei = complex<T>(0, 0);
        Ej1 = complex<T>(0, 0);
        Ej2 = complex<T>(0, 0);
        Ej1j2 = complex<T>(0, 0);
        Ej1k1 = complex<T>(0, 0);
        Ej2k2 = complex<T>(0, 0);

        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * (U0 + dU[i]) * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                Ej1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                Ej2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    Ej1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                    Ej2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
                }
                if (n < nmax - 1) {
                    Ej1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                    Ej2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
                }

                if (n > 1) {
                    Ej1 += -J[j1] * J[j1] * exp2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                    Ej2 += -J[i] * J[i] * expm2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
                }
                if (n < nmax - 2) {
                    Ej1 -= -J[j1] * J[j1] * exp2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                    Ej2 -= -J[i] * J[i] * expm2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j2, n, n + 3, i, j2, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j2][n + 1] * f[i][n] * f[j2][n + 3];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        Ej1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                                ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        Ej2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                                ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                        Ej1 += J[j1] * expth * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        Ej2 += J[i] * expmth * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                        if (n != m - 3 && m > 1 && n < nmax - 1) {
                            Ej1 += -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 += -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                        if (n != m + 1 && n > 0 && m < nmax) {
                            Ej1 -= -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 -= -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }

                        if (n > 0) {
                            Ej1j2 += -J[j1] * J[i] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j1, n, m, i, j2, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][n - 1]
                                    * f[i][n - 1] * f[j1][m] * f[j2][n];
                            Ej1j2 += -J[i] * J[j1] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j2, n, m, i, j1, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][n - 1]
                                    * f[i][n - 1] * f[j2][m] * f[j1][n];
                        }
                        if (n < nmax - 1) {
                            Ej1j2 -= -J[j1] * J[i] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j1, n, m, i, j2, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][n + 1]
                                    * f[i][n] * f[j1][m] * f[j2][n + 2];
                            Ej1j2 -= -J[i] * J[j1] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j2, n, m, i, j1, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][n + 1]
                                    * f[i][n] * f[j2][m] * f[j1][n + 2];
                        }

                        Ej1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m] -
                                ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]);
                        Ej2 += -0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m] -
                                ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]);

                        for (int q = 1; q <= nmax; q++) {
                            if (n < nmax - 1 && n != q - 2) {
                                Ej1j2 += -0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n] * f[j1][m] * f[j2][q];
                                Ej1j2 += -0.5 * J[i] * J[j1] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n] * f[j2][m] * f[j1][q];
                            }
                            if (n > 0 && n != q) {
                                Ej1j2 -= -0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n - 1] * f[j1][m] * f[j2][q];
                                Ej1j2 -= -0.5 * J[i] * J[j1] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n - 1] * f[j2][m] * f[j1][q];
                            }

                            if (m != q) {
                                Ej1k1 += -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m] * f[k1][q];
                                Ej2k2 += -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m] * f[k2][q];
                                Ej1k1 -= -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m - 1] * f[k1][q];
                                Ej2k2 -= -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m - 1] * f[k2][q];
                            }

                        }

                        for (int p = 0; p < nmax; p++) {

                            if (p != n - 1 && 2 * n - m == p && n > 0) {
                                Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                                Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != n - 1 && 2 * n - m != p && n > 0) {
                                Ej1j2 += -0.25 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 += -0.25 * J[i] * J[j1] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m != p - 2 && n < nmax - 1) {
                                Ej1j2 -= -0.25 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 -= -0.25 * J[i] * J[j1] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != m - 1 && n != p) {
                                Ej1k1 += -0.25 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                        ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                                Ej2k2 += -0.25 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                        ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                            }
                        }

                        Ej1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                                * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                                * f[i][n] * f[j1][m] * f[k1][n + 1]);
                        Ej2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                                * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                                * f[i][n] * f[j2][m] * f[k2][n + 1]);

                        Ej1k1 += -J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                        Ej2k2 += -J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                        if (m != n - 1 && n != m && m < nmax && n > 0) {
                            Ej1 += -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 += -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }
                        if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                            Ej1 -= -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 -= -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                    }
                }
            }
        }

        Ei /= norm2[i];
        Ej1 /= norm2[i] * norm2[j1];
        Ej2 /= norm2[i] * norm2[j2];
        Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
        Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
        Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

        E += Ei;
        E += Ej1;
        E += Ej2;
        E += Ej1j2;
        E += Ej1k1;
        E += Ej2k2;
    }

    return E.real();
}

template<class T>
T energy(CppAD::vector<T>& fin) {//, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {
    return energy(fin, 0);
}

typedef CppAD::AD<complex<double>> ADc;

using namespace CppAD;

class FG_eval {
public:
    typedef CppAD::vector<CppAD::AD<double>> ADvector;
    
    FG_eval(double theta_) : theta(theta_) {}

    void operator()(ADvector& fg, const ADvector& x) {
        ptime begin = microsec_clock::local_time();
//        cout << x << endl;
        fg[0] = energy(const_cast<ADvector&> (x), theta);
//        cout << fg[0] << endl;
        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
//        cout << endl << period.length() << endl << endl;
        for (int i = 0; i < L; i++) {
            //                fg[i+1] = norm(i, const_cast<ADvector&>(x));
        }
    }
    
private:
    double theta;
};

class thread_id {
public:
    int id;

    thread_id(int i) : id(i) {
    }
};

boost::thread_specific_ptr<thread_id> tls;

void thread_func(int i, double theta) {
    tls.reset(new thread_id(i));

    typedef CppAD::vector<CppAD::AD<double>> ADvector;
    typedef CppAD::vector<double> Dvector;

    // number of independent variables (domain dimension for f and g)
    size_t nx = 2 * L*dim;
    // number of constraints (range dimension for g)
    size_t ng = 0; //L;
    // initial value of the independent variables
    //    Dvector xi(nx, 1.0);
    Dvector xi(nx);
    for (int i = 0; i < nx; i++) {
        xi[i] = 1.0;
    }
    Dvector xl(nx), xu(nx);
    for (int i = 0; i < nx; i++) {
        xl[i] = -2.0;
        xu[i] = 2.0;
    }
    // lower and upper limits for g
    Dvector gl(ng), gu(ng);
    for (int i = 0; i < ng; i++) {
        gl[i] = 1.0;
        gu[i] = 1.0;
    }

    // options 
    std::string options;
    // turn off any printing
    	options += "Integer print_level  0\n"; 
    options += "String  sb           yes\n";
    // maximum number of iterations
    options += "Integer max_iter     4000\n";
    // approximate accuracy in first order necessary conditions;
    // see Mathematical Programming, Volume 106, Number 1, 
    // Pages 25-57, Equation (6)
    options += "Numeric tol          1e-12\n";
    options += "Numeric acceptable_tol          1e-12\n";
    // derivative testing
//    	options += "String  derivative_test            second-order\n";
    // maximum amount of random pertubation; e.g., 
    // when evaluation finite diff
//    options += "Numeric point_perturbation_radius  0.\n";

//        options += "String hessian_approximation limited-memory\n";
    //    options += "String hessian_approximation exact\n";

    options += "String linear_solver ma86\n";
    options += "Sparse true reverse\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // object that computes objective and constraints
    FG_eval fg_eval(theta);

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, xi,
            xl, xu, gl, gu, fg_eval, solution
            );

    cout << solution.status << endl;
    cout << "E(" << theta << ") = " << lexical_cast<string>(solution.obj_value) << endl;
}

void thread_func(int i) {
    tls.reset(new thread_id(i));

    typedef CppAD::vector<CppAD::AD<double>> ADvector;
    typedef CppAD::vector<double> Dvector;

    // number of independent variables (domain dimension for f and g)
    size_t nx = 2 * L*dim;
    // number of constraints (range dimension for g)
    size_t ng = 0; //L;
    // initial value of the independent variables
    //    Dvector xi(nx, 1.0);
    Dvector xi(nx);
    for (int i = 0; i < nx; i++) {
        xi[i] = 1.0;
    }
    Dvector xl(nx), xu(nx);
    for (int i = 0; i < nx; i++) {
        xl[i] = -2.0;
        xu[i] = 2.0;
    }
    // lower and upper limits for g
    Dvector gl(ng), gu(ng);
    for (int i = 0; i < ng; i++) {
        gl[i] = 1.0;
        gu[i] = 1.0;
    }

    // options 
    std::string options;
    // turn off any printing
    	options += "Integer print_level  0\n"; 
    options += "String  sb           yes\n";
    // maximum number of iterations
    options += "Integer max_iter     2000\n";
    // approximate accuracy in first order necessary conditions;
    // see Mathematical Programming, Volume 106, Number 1, 
    // Pages 25-57, Equation (6)
    options += "Numeric tol          1e-12\n";
    options += "Numeric acceptable_tol          1e-12\n";
    // derivative testing
//    	options += "String  derivative_test            second-order\n";
    // maximum amount of random pertubation; e.g., 
    // when evaluation finite diff
//    options += "Numeric point_perturbation_radius  0.\n";

        options += "String hessian_approximation limited-memory\n";
    //    options += "String hessian_approximation exact\n";

    options += "String linear_solver ma86\n";
    options += "Sparse true reverse\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // object that computes objective and constraints
    FG_eval fg_eval0(0);

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, xi,
            xl, xu, gl, gu, fg_eval0, solution
            );

//    Dvector norms(L);
//    for (int i = 0; i < L; i++) {
//        norms[i] = abs(i, solution.x);
//    }
//    Dvector xnorm(2 * L * dim);
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//            xnorm[2 * (i * dim + n)] = solution.x[2 * (i * dim + n)] / norms[i];
//            xnorm[2 * (i * dim + n) + 1] = solution.x[2 * (i * dim + n) + 1] / norms[i];
//        }
//    }
    cout << solution.status << endl;
    cout << "E0 = " << lexical_cast<string>(solution.obj_value) << endl;

    // object that computes objective and constraints
    FG_eval fg_evalth(0.01);

    for (int i = 0; i < nx; i++) {
        xi[i] = 1.0;
    }
    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, xi,
            xl, xu, gl, gu, fg_evalth, solution
            );

    cout << solution.status << endl;
    cout << "Eth = " << lexical_cast<string>(solution.obj_value) << endl;
}

bool parallel = false;

bool in_parallel() {
    return parallel;
}

size_t thread_num() {
    return tls->id;
}

double energyfunc(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    CppAD::ADFun<double>* func = static_cast<CppAD::ADFun<double>*> (data);
    CppAD::vector<double> cppx(x.size());
    for (int i = 0; i < x.size(); i++) {
        cppx[i] = x[i];
    }
    CppAD::vector<double> cppgrad = func->Jacobian(cppx);
    copy(cppgrad.data(), cppgrad.data() + grad.size(), grad.begin());
    cout << func->Forward(0, cppx)[0] << endl;
    return func->Forward(0, cppx)[0];
}

void thread_func2(int i) {
    tls.reset(new thread_id(i));
    ptime begin, end;

    CppAD::vector<CppAD::AD<double>> qw(2 * L * dim);
    for (int i = 0; i < 2 * L * dim; i++) {
        qw[i] = 1;
    }
    CppAD::Independent(qw);
    FG_eval fge(0);
    CppAD::vector<CppAD::AD<double>> as(1);
    begin = microsec_clock::local_time();
    //    fge(as, qw);
    as[0] = energy(qw);
    CppAD::ADFun<double> zx(qw, as);
    zx.optimize();
    end = microsec_clock::local_time();
    time_period period1(begin, end);
    cout << endl << period1.length() << endl << endl;

    int ndim = 2 * L*dim;
    nlopt::opt lopt(nlopt::algorithm::LD_LBFGS, ndim);
    lopt.set_lower_bounds(-2);
    lopt.set_upper_bounds(2);
    lopt.set_min_objective(energyfunc, &zx);
    lopt.set_ftol_rel(1e-16);
    lopt.set_ftol_abs(1e-16);

    double E0;
    std::vector<double> x(ndim, 1);
    begin = microsec_clock::local_time();
    lopt.optimize(x, E0);
    end = microsec_clock::local_time();
    time_period period4(begin, end);
    cout << endl << period4.length() << endl << endl;
    cout << "E0(" << i << ") = " << E0 << endl;
}

int roundUp(int numToRound, int multiple) {
    if (multiple == 0) {
        return numToRound;
    }

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;
    return numToRound + multiple - remainder;
}

lbfgsfloatval_t lbfgs_eval(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {
//    int nx = roundUp(2 * L*dim, 16);
    CppAD::ADFun<double>* func = static_cast<CppAD::ADFun<double>*> (instance);
    CppAD::vector<double> cppx(2 * L * dim);
    for (int i = 0; i < 2 * L * dim; i++) {
        cppx[i] = x[i];
    }
    CppAD::vector<double> cppgrad = func->Jacobian(cppx);
//    fill(g, g + nx, 0);
    copy(cppgrad.data(), cppgrad.data() + 2 * L*dim, g);
    return func->Forward(0, cppx)[0];

}

/*
 * 
 */
int main(int argc, char** argv) {

    cout << setprecision(16);
    
    /*CppAD::vector<CppAD::AD<double>> qw(2*L*dim);
    for(int i = 0; i < 2*L*dim; i++) {
        qw[i] = 1;//xc[i];
    }
    CppAD::Independent(qw);
    CppAD::vector<CppAD::AD<double>> as(1);
        as[0] = energy(qw);
    CppAD::ADFun<double> zx(qw,as);
    zx.optimize();

    int rnx = 2*L*dim;//roundUp(2 * L*dim, 16);
    lbfgsfloatval_t *lx = lbfgs_malloc(rnx);
    lbfgs_parameter_t param;
    for(int i = 0; i < 2*L*dim; i++) {
        lx[i] = 1;
    }
//    for(int i = 2*L*dim; i < rnx; i++) {
//        lx[i] = 0;
//    }
    lbfgs_parameter_init(&param);
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    param.epsilon = 1e-12;
    lbfgsfloatval_t fx;
    int ret = lbfgs(rnx, lx, &fx, lbfgs_eval, NULL, &zx, &param);
    std::vector<double> xv(2*L*dim);
    copy(lx, lx+2*L*dim, xv.data());
    std::vector<double> xvnorms(L);
    for (int i = 0; i < L; i++) {
        xvnorms[i] = abs(i, xv);
    }
    std::vector<double> xvnorm(2 * L * dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            xvnorm[2 * (i * dim + n)] = xv[2 * (i * dim + n)] / xvnorms[i];
            xvnorm[2 * (i * dim + n) + 1] = xv[2 * (i * dim + n) + 1] / xvnorms[i];
        }
    }
    for(int i = 0; i < 2*L*dim; i++) {
        cout << lexical_cast<string>(xvnorm[i]) << endl;
    }
    cout << lexical_cast<string>(fx) << endl;
    cout << "ret = " << ret << endl;
    lbfgs_free(lx);
    exit(0);*/

    /*std::vector<double> fin(2*L*dim, 0);
    for(int i = 0; i < L; i++) {
        fin[2*i*dim+2] = 1;
    }
   CppAD::vector<double> finad;
   finad.push_vector(fin);
   double E0 = energy(finad);
   cout << lexical_cast<std::string>(E0) << endl;
   exit(0);*/

    /*tls.reset(new thread_id(0));
    thread_alloc::parallel_setup(3, in_parallel, thread_num);
    thread_alloc::hold_memory(true);
    CppAD::parallel_ad<double>();
    CppAD::CheckSimpleVector<size_t, CppAD::vector<size_t>>();
    CppAD::CheckSimpleVector<set<size_t>, CppAD::vector<set<size_t>>>(CppAD::one_element_std_set<size_t>(), CppAD::two_element_std_set<size_t>());
    parallel = true;
//    CppAD::RevSparseJacSet()
    
    thread_group threads2;
    for (int i = 0; i < 2; i++) {
        threads2.create_thread(boost::bind(thread_func2, i+1));
    }
    threads2.join_all();
    exit(0);

        ptime begin, end;

    CppAD::vector<CppAD::AD<double>> qw(2*L*dim);
    for(int i = 0; i < 2*L*dim; i++) {
        qw[i] = 1;//xc[i];
    }
    CppAD::Independent(qw);
    FG_eval fge;
    CppAD::vector<CppAD::AD<double>> as(1);
        begin = microsec_clock::local_time();
//    fge(as, qw);
        as[0] = energy(qw);
    CppAD::ADFun<double> zx(qw,as);
    zx.optimize();
        end = microsec_clock::local_time();
        time_period period1(begin, end);
        cout << endl << period1.length() << endl << endl;
    
    int ndim = 2*L*dim;
    nlopt::opt lopt(nlopt::algorithm::LD_LBFGS, ndim);
        lopt.set_lower_bounds(-2);
        lopt.set_upper_bounds(2);
        lopt.set_min_objective(energyfunc, &zx);
        lopt.set_ftol_rel(1e-16);
        lopt.set_ftol_abs(1e-16);
        
        double E0;
        std::vector<double> x(ndim, 1);
        begin = microsec_clock::local_time();
        lopt.optimize(x, E0);
        end = microsec_clock::local_time();
        time_period period4(begin, end);
        cout << endl << period4.length() << endl << endl;
        cout << "E0 = " << E0 << endl;
//        for (int i = 0; i < ndim; i++) {
//        cout << x[i] << " ";
//        }
//        cout << endl;

    exit(0);

    CppAD::vector<double> we(2*L*dim);
    for(int i = 0; i < 2*L*dim; i++) {
        we[i] = 1;
    }
        begin = microsec_clock::local_time();
    zx.Forward(0, we);
        end = microsec_clock::local_time();
        time_period period2(begin, end);
        cout << endl << period2.length() << endl << endl;
        begin = microsec_clock::local_time();
    CppAD::vector<double> sd = zx.Jacobian(we);
        end = microsec_clock::local_time();
        time_period period3(begin, end);
        cout << endl << period3.length() << endl << endl;
    exit(0);*/

    tls.reset(new thread_id(0));
    thread_alloc::parallel_setup(3, in_parallel, thread_num);
    thread_alloc::hold_memory(true);
    CppAD::parallel_ad<double>();
    CppAD::CheckSimpleVector<size_t, CppAD::vector < size_t >> ();
    CppAD::CheckSimpleVector<set<size_t>, CppAD::vector<set < size_t>>>(CppAD::one_element_std_set<size_t>(), CppAD::two_element_std_set<size_t>());
    parallel = true;
    //    CppAD::RevSparseJacSet()

    thread_group threads;
    for (int i = 0; i < 1; i++) {
//        threads.create_thread(boost::bind(thread_func, i + 1));
    }
        threads.create_thread(boost::bind(thread_func, 1, 0));
        threads.create_thread(boost::bind(thread_func, 2, 0.01));
    threads.join_all();
    exit(0);

    typedef std::vector<CppAD::AD<double>> ADvector;
    typedef std::vector<double> Dvector;

    // number of independent variables (domain dimension for f and g)
    size_t nx = 2 * L*dim;
    // number of constraints (range dimension for g)
    size_t ng = 0; //L;
    // initial value of the independent variables
    Dvector xi(nx, 1.0);
    // lower and upper limits for x
    Dvector xl(nx), xu(nx);
    for (int i = 0; i < nx; i++) {
        xl[i] = -2.0;
        xu[i] = 2.0;
    }
    // lower and upper limits for g
    Dvector gl(ng), gu(ng);
    for (int i = 0; i < ng; i++) {
        gl[i] = 1.0;
        gu[i] = 1.0;
    }

    // object that computes objective and constraints
    FG_eval fg_eval(0);
    //    ADvector zxc(2*L*dim, 1);
    //    Independent(zxc);
    //    ADvector qwe(L+1);
    //    fg_eval(qwe, zxc);
    ////    AD<double> asd = qwe[0];
    //    ADvector asd({qwe[0]});
    //    ADFun<double> fun(zxc, asd);
    //    Dvector zxc2(2*L*dim, 1);
    //    Dvector hess(2*L*dim*2*L*dim);
    //    cout << "About to Hessian" << endl;
    //    hess = fun.Hessian(zxc2, 0);
    //    cout << "Hessianed" << endl;
    //    int count = 0;
    //    for(int i = 0; i < 2*L*dim*2*L*dim; i++) {
    ////        cout << jac[i] << " ";
    //        if(NearEqual(0., hess[i], 1e-10, 1e-10)) {
    //            count++;
    //        }
    //    }
    //    cout << "Zero count: " << count << endl;
    //    exit(0);

    //	// number of independent variables (domain dimension for f and g)
    //	size_t nx = 4;  
    //	// number of constraints (range dimension for g)
    //	size_t ng = 2;
    //	// initial value of the independent variables
    //	Dvector xi(nx);
    //	xi[0] = 1.0;
    //	xi[1] = 5.0;
    //	xi[2] = 5.0;
    //	xi[3] = 1.0;
    //	// lower and upper limits for x
    //	Dvector xl(nx), xu(nx);
    //	for(int i = 0; i < nx; i++)
    //	{	xl[i] = 1.0;
    //		xu[i] = 5.0;
    //	}
    //	// lower and upper limits for g
    //	Dvector gl(ng), gu(ng);
    //	gl[0] = 25.0;     gu[0] = 1.0e19;
    //  	gl[1] = 40.0;     gu[1] = 40.0;
    //
    //	// object that computes objective and constraints
    //	FG_eval fg_eval;

    // options 
    std::string options;
    // turn off any printing
    //	options += "Integer print_level  0\n"; 
    options += "String  sb           yes\n";
    // maximum number of iterations
    options += "Integer max_iter     1000\n";
    // approximate accuracy in first order necessary conditions;
    // see Mathematical Programming, Volume 106, Number 1, 
    // Pages 25-57, Equation (6)
    options += "Numeric tol          1e-12\n";
    options += "Numeric acceptable_tol          1e-12\n";
    // derivative testing
    //	options += "String  derivative_test            second-order\n";
    // maximum amount of random pertubation; e.g., 
    // when evaluation finite diff
    options += "Numeric point_perturbation_radius  0.\n";

    //    options += "String hessian_approximation limited-memory\n";
    //    options += "String hessian_approximation exact\n";

    options += "String linear_solver ma97\n";
    options += "Sparse true reverse\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, xi, xl, xu, gl, gu, fg_eval, solution
            );

    for (int i = 0; i < 2 * dim; i++) {
        cout << solution.x[i] << " ";
    }
    cout << endl;
    //    cout << solution.x << endl;
    cout << solution.obj_value << endl;
    cout << solution.status << endl;
    for (int i = 0; i < L; i++) {
        //        cout << norm(i, solution.x) << endl;
    }

    std::vector<double> norms(L);
    for (int i = 0; i < L; i++) {
        norms[i] = abs(i, solution.x);
    }
    //    std::vector<CppAD::AD<double>> xnorm(2*L*dim);
    ADvector xnorm(2 * L * dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            xnorm[2 * (i * dim + n)] = solution.x[2 * (i * dim + n)] / norms[i];
            xnorm[2 * (i * dim + n) + 1] = solution.x[2 * (i * dim + n) + 1] / norms[i];
        }
    }
    ADvector fg(L + 1);
    //    fg_eval(fg, xnorm);
    //    cout << fg[0] << endl;
    for (int i = 0; i < L; i++) {
        //        cout << norm(i, xnorm) << endl;
    }

    /*using namespace CppAD;
    using namespace CppAD::cg;
    
//    typedef AD<complex<double>> ADC;
    typedef AD<double> ADC;
    std::vector<ADC> x(2);
    x[0] = 4;
    x[1] = 10;
    Independent(x);
    cout << x[0] << endl;
    
    std::vector<ADC> y(2);
    y[0] = x[0];
    y[1] = 0.5*x[1];
    cout << y[1] << endl;
    
    ADFun<double> fun(x, y);*/



    //    AD<double> x;
    //    AD<double> y = 0.5*x;

    ////    typedef CppAD::cg::CG<complex<double>> CGD;
    //    typedef CppAD::cg::CG<double> CGD;
    //    typedef CppAD::AD<CGD> ADCG;
    //
    //    std::vector<ADCG> X(2);
    //    CppAD::Independent(X);
    //
    //    // dependent variable vector 
    //    std::vector<ADCG> Y(1);
    //
    //    // the model
    //    ADCG a = X[0] / 1. + X[1] * X[1];
    //    Y[0] = a / 2;
    //
    //    CppAD::ADFun<CGD> fun(X, Y);
    //    
    //    ModelCSourceGen<double> cgen(fun, "model");
    //    cgen.setCreateJacobian(true);
    //    cgen.setCreateForwardOne(true);
    //    cgen.setCreateReverseOne(true);
    //    cgen.setCreateReverseTwo(true);
    //    ModelLibraryCSourceGen<double> libcgen(cgen);
    //
    //    // compile source code
    //    DynamicModelLibraryProcessor<double> p(libcgen);
    //
    //    GccCompiler<double> compiler;
    //    DynamicLib<double>* dynamicLib = p.createDynamicLibrary(compiler);
    //    
    //    // save to files (not really required)
    //    SaveFilesModelLibraryProcessor<double> p2(libcgen);
    //    p2.saveSources();

    return 0;
}

