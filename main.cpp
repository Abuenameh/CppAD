/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 15 August 2015, 18:16
 */

#define CPPAD_CG_SYSTEM_LINUX 1

#include <cstdlib>
#include <vector>

using namespace std;

#include <boost/date_time.hpp>
using namespace boost::posix_time;


//#include <cppad/cg/cppadcg.hpp>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
//#include <cppad/ipopt/solve_callback.hpp>
//using namespace CppAD;
using CppAD::AD;

typedef CppAD::AD<double> Scalar;

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
operator*(const complex<CppAD::AD<_Tp>>& __x, const _Tp& __y)
{
    complex<CppAD::AD<_Tp>> __t(__x);
    __t *= CppAD::AD<_Tp>(__y);
    return __t;
}

template<class _Tp>
inline
complex<CppAD::AD<_Tp>>
operator*(const _Tp& __x, const complex<CppAD::AD<_Tp>>& __y)
{
    complex<CppAD::AD<_Tp>> __t(__y);
    __t *= CppAD::AD<_Tp>(__x);
    return __t;
}


inline int mod(int i) {
	return (i + L) % L;
}

inline double g2(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline double eps(vector<double>& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}

inline Scalar g(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

//inline Scalar eps(vector<double>& U, int i, int j, int n, int m) {
//	return n * U[i] - (m - 1) * U[j];
//}
//
//inline Scalar eps(double U, int n, int m) {
//    return (n - m + 1) * U;
//}

inline Scalar eps(vector<double>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n*U[i] - (m-1)*U[j] + (q-1)*U[k] - p*U[l];
}

AD<double> norm(int i, vector<AD<double>>& fin) {

    complex<AD<double>>* f;
    AD<double> norm2 = 0;
        f = reinterpret_cast<complex<AD<double>>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2 += norm(f[n]);//f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
        }
        return norm2;
}

double norm(int i, vector<double>& fin) {

    complex<double>* f;
    double norm2 = 0;
        f = reinterpret_cast<complex<double>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2 += norm(f[n]);//f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
        }
        return norm2;
}

double abs(int i, vector<double>& fin) {

    complex<double>* f;
    double norm2 = 0;
        f = reinterpret_cast<complex<double>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2 += norm(f[n]);//f[m].real() * f[m].real() + f[m].imag() * f[m].imag();
        }
        return sqrt(norm2);
}

AD<double> energy(/*int i, int n,*/ vector<AD<double>>& fin) {//, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {

//    vector<double> J(L, 0.2);
    double U0 = 1;
//    vector<double> dU(L, 0);//.01);
//    double mu = 0.5;
    double theta = 0;
    
//    vector<double> dU({-0.062495818684028,-0.240152540894773,-0.245433857301291,-0.273625816197061,-0.242270144341017,-0.27259078540955,-0.20204475712817,-0.0848398286345957,0.424865322680541,-0.0971718211590916,-0.129687170384589,-0.271703531802801,0.0948468516335159,-0.117917878473979,-0.0135564488854456,0.419054844730518,-0.108478764896881,-0.216038019921792,0.260931974004009,-0.202439024332201,0.187098382438594,-0.183422731248427,-0.146060037885631,-0.156016398905386,-0.183701971564288});
//    vector<double> J({0.138152396495495,0.140219774273089,0.140581397909088,0.140546956950688,0.140535853601678,0.140093899853519,0.138001669064538,0.129914946537351,0.13004950249811,0.137341045165717,0.139269569050823,0.136554490078846,0.134896150878805,0.136228606430583,0.129211199655538,0.130262768487484,0.138427531129939,0.133745485153161,0.13360220926095,0.134615642880126,0.134412331368716,0.138498071670926,0.138193037947643,0.138613245204837,0.137539881531959});
//    double mu = 0.26669999999999999263;
    
    vector<double> dU({0.043199239962729896,0.08259186491718995,0.19397213987527007,0.31412171808007994,
    0.0915189876615099,0.32699155669771995,0.03970221015499997,0.3169288596880999,
    -0.06649698521376701,0.11030126264836992,0.13055555283211007,-0.10016340121284895,
    -0.054463258625564004,-0.17320924136983595,0.35891542195234005,-0.364378648638469,
    0.4271911797647201,-0.19375941242682504,-0.100964667952118,-0.019592374277792035,0.26492282856173,
    0.28402138986372005,0.025510001447129893,-0.017569415902372043,0.060356175879900054,
    -0.09299207573504398,0.39096937992084,0.30643104688444,-0.353520812761554,-0.139929321387956,
    -0.34123939491588895,0.13262668956456003,-0.391729203944317,-0.113887642455067,
    0.30318444229128994,0.42098839070455996,0.2522801656999001,-0.30002135151780596,0.33836311285219,
    0.33843394927349,0.44146187686278004,-0.023139212960445965,0.27186045802215997,
    0.27349689504697006,-0.03372008677117999,0.01805996653825992,0.25448881206848006,
    0.1606392258635101,-0.31722957528124096,0.19898802140462002});
    vector<double> J({0.0199872024688441,0.0214962626425955,0.0238803449658659,0.0227504318202043,
    0.0228723747345574,0.0222827981052472,0.0221899440052608,0.0209501537515912,0.0190945600203537,
    0.021156522094292,0.0189093575813226,0.0171830354126276,0.0164309269687668,0.0199880266581662,
    0.0174175184250734,0.0178953155275678,0.0202658069422593,0.0157860325229642,0.017509454115344,
    0.021035343501827,0.0243318084090072,0.0217241042749188,0.0188015915058845,0.0191413581434165,
    0.0183451209145919,0.0212626001654235,0.0258778593153581,0.017195760116303,0.0137369700547507,
    0.0138722624661954,0.0160682496382933,0.0154155045908814,0.0135204219040843,0.020262193793736,
    0.0261493969212978,0.0255861770198534,0.0175138263853036,0.018162793198886,0.0256728733632031,
    0.0267461529034936,0.0225589323559612,0.0210575213835723,0.0242950606632568,0.0209503493374639,
    0.0185636023452844,0.0213662921171238,0.0229271322131208,0.016588609400904,0.0168834475170639,
    0.0211207531674717});
    double mu = 0.9;
    
    complex<AD<double>> expth = complex<AD<double>>(cos(theta), sin(theta));
    complex<AD<double>> expmth = ~expth;
    complex<AD<double>> exp2th = expth*expth;
    complex<AD<double>> expm2th = ~exp2th;

    vector<complex<Scalar>* > f(L);
    vector<Scalar> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        f[j] = reinterpret_cast<complex<Scalar>*> (&fin[2 * j * dim]);
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += norm(f[j][m]);//f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<AD<double>> E = complex<AD<double>>(0, 0);

    complex<AD<double>> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

        for (int i = 0; i < L; i++) {

    int k1 = mod(i - 2);
    int j1 = mod(i - 1);
    int j2 = mod(i + 1);
    int k2 = mod(i + 2);

    Ei = complex<AD<double>>(0, 0);
    Ej1 = complex<AD<double>>(0, 0);
    Ej2 = complex<AD<double>>(0, 0);
    Ej1j2 = complex<AD<double>>(0, 0);
    Ej1k1 = complex<AD<double>>(0, 0);
    Ej2k2 = complex<AD<double>>(0, 0);

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

//    Ei /= norm2[i];
//    Ej1 /= norm2[i] * norm2[j1];
//    Ej2 /= norm2[i] * norm2[j2];
//    Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
//    Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
//    Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

    E += Ei;
    E += Ej1;
    E += Ej2;
    E += Ej1j2;
    E += Ej1k1;
    E += Ej2k2;
        }

    return E.real();
}

typedef CppAD::AD<complex<double>> ADc;

        using namespace CppAD;
	class FG_eval {
	public:
        typedef std::vector<CppAD::AD<double>> ADvector;
		void operator()(ADvector& fg, const ADvector& x)
		{	
        ptime begin = microsec_clock::local_time();
            fg[0] = energy(const_cast<ADvector&>(x));
        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;
            for(int i = 0; i < L; i++)
            {
                fg[i+1] = norm(i, const_cast<ADvector&>(x));
            }
		}
	};

/*
 * 
 */
int main(int argc, char** argv) {
    
    cout << setprecision(16);
    
        typedef std::vector<CppAD::AD<double>> ADvector;
    typedef std::vector<double> Dvector;

	// number of independent variables (domain dimension for f and g)
	size_t nx = 2*L*dim;  
	// number of constraints (range dimension for g)
	size_t ng = L;
	// initial value of the independent variables
	Dvector xi(nx, 1.0);
//    Dvector xi({0.0429308663421594,0.0471336276897605,0.703723190543898,0.703723190543898,0.0547415917565458,
//   0.0498602349003013,1.28627160371287e-6,9.68955206990986e-7,-3.73786273248352e-10,
//   -2.19486927672599e-10,4.98369073687976e-9,-5.9782524264129e-11,0.0508847580502529,
//   0.0547568480015739,0.699062220772132,0.699062220772132,0.0956066682073325,0.0888458906599719,
//   0.00002126011946267,0.000017036730691444,1.73952208726951e-7,1.29234576072524e-7,
//   1.05219975255779e-9,7.08473034254255e-10,0.0620299237940634,0.0654261126663519,0.694450223597663,
//   0.694450223597663,0.120004554973573,0.113775272964817,0.000036233905785684,0.0000308609760040107,
//   3.67503003598888e-7,2.96490329318757e-7,3.02104193316953e-9,2.35526287088823e-9,0.0659883696200746,
//   0.0682219827102214,0.691275346039436,0.691275346039436,0.134964055942492,0.13054528084119,
//   0.0000658748715198576,0.0000596052118594174,8.37533578271024e-7,7.32847854139964e-7,
//   8.04869010873389e-9,6.85664022987026e-9,0.0676552192773875,0.0685600083756981,0.692697366241488,
//   0.692697366241488,0.125442269024305,0.123786798492041,0.000039970118247513,0.0000384079840957729,
//   4.14167724747042e-7,3.92719516308242e-7,3.67904314102964e-9,3.33452520125732e-9,0.058242188967717,
//   0.0578525983262944,0.695916964829232,0.695916964829232,0.11065477104127,0.111400001266891,
//   0.000037240306676699,0.0000379992728047351,3.90934404612453e-7,4.0160721545836e-7,
//   3.07649056672569e-9,3.19607300628788e-9,0.0485800377198082,0.0472997654401655,0.701672753514069,
//   0.701672753514069,0.0722046030814904,0.0741592817119779,6.43697121727515e-6,6.97414977681334e-6,
//   3.31261327574098e-8,3.68577949904538e-8,2.00168899267957e-10,1.97888105194715e-10,
//   0.0352172055672912,0.0336104862638931,0.705410722637478,0.705410722637478,0.033973978689574,
//   0.0355988322444576,3.71204388033982e-7,4.26513427887271e-7,2.36631852526141e-10,
//   2.92345616823172e-10,-1.57331785673076e-9,-1.7316482344609e-10,0.0205043131054873,
//   0.0208719795277999,0.676352651230666,0.735946503259928,0.00559789716694676,0.00651400070345999,
//   3.60854028481249e-9,4.67750457076586e-9,1.67893842260655e-10,-2.63016853258532e-10,
//   -1.26001029416464e-10,1.86779768841348e-10,0.00924602799709112,0.00848203101314742,
//   0.707027051838049,0.707027051838049,0.00557716789010993,0.0060792405017451,5.4819003063968e-9,
//   7.30981871472939e-9,-9.49753208745056e-12,-1.37376052585295e-11,-3.06037738379914e-9,
//   -1.03367861898577e-9,0.0055789290420063,0.0050210942927914,0.707071651429895,0.707071651429895,
//   0.00438774650638828,0.00487558519608827,3.42626059130933e-10,1.48131320527982e-9,
//   -3.81240709764077e-11,-5.4622961223219e-11,-7.82133993555624e-10,-5.83468598876132e-10,
//   0.00335886215794214,0.00296930934035872,0.707087020972924,0.707087020972924,0.00396006210429946,
//   0.00448426943062411,1.19361537942316e-9,3.00543334038755e-9,-4.65626576278655e-12,
//   -2.32084164017634e-11,-2.41611901241474e-11,-2.15160637734879e-11,0.00256579996304252,
//   0.00223619727838832,0.707101045531351,0.707101045531351,0.00141110920818306,0.00162716910060883,
//   1.04684570253241e-10,1.45992904011811e-10,-1.38951212676784e-11,-1.11075342207737e-11,
//   -1.41260195632493e-11,-8.4623760284245e-10,0.00131978586164036,0.0011529551198201,
//   0.707104976189403,0.707104976189403,0.000929648027888225,0.00108162406612092,-8.28832261407854e-10,
//   -8.63032156273584e-10,-1.0905250795359e-11,-1.09027543828392e-11,2.81918943679353e-10,
//   1.82623325130368e-10,0.00086441400959447,0.000787097635579605,0.707106168736172,0.707106168736172,
//   0.000396806883547736,0.000456158451498258,-1.12031416772684e-10,-6.28870488665374e-11,
//   -7.92307294478525e-11,-4.93838823016623e-11,-7.25703739797776e-10,-4.81941546862715e-9,
//   0.000702464543887347,0.000731505774725777,0.707106411244964,0.707106411244964,
//   0.0000983918192746323,0.0000900802456933566,8.07432185985916e-12,-2.24351401840652e-11,
//   -2.72891481006645e-9,-7.64496784712961e-9,7.61900657923034e-11,-1.02028811739366e-10,
//   0.000691490155566792,0.000817770784460533,0.707106190930912,0.707106190930912,0.000562492081149383,
//   0.000454080742777565,1.0921691892584e-11,-3.86128642827382e-10,-9.98108871861073e-11,
//   -2.47062845585218e-11,-6.24714950211632e-10,-5.68161797847627e-10,0.000898466282923141,
//   0.00110971655882527,0.707105391837824,0.707105391837824,0.00107538019293169,0.000857037090021562,
//   2.31275216680701e-9,2.0298063287142e-9,-1.74237713171017e-11,-1.21685972991633e-11,
//   1.6522669233877e-11,9.49312129199428e-11,0.00183846027549716,0.00229505184261752,0.707103204342174,
//   0.707103204342174,0.000948670439094939,0.000754747007146614,6.66014417280751e-11,
//   -2.11901528388701e-11,-2.05167611291681e-10,-3.92459080747983e-11,-4.24316243499393e-11,
//   -1.28773018266013e-10,0.00271207324517588,0.00334746974196639,0.707092419280545,0.707092419280545,
//   0.00365249425947661,0.00295289119466671,2.45441472525392e-9,1.57459601052357e-9,
//   -1.8820076552597e-11,-9.14621284329064e-11,-4.0538298522093e-12,-7.06741234043776e-12,
//   0.00798583619343773,0.00508169722552062,0.89120239859545,0.45347179598755,0.00526661292702802,
//   0.00206893772405547,3.15556116160715e-9,6.45745855850083e-10,9.86198664384965e-9,
//   -7.43544005960599e-9,-1.93127872991931e-10,-3.6351459218381e-10,0.00897357169353097,
//   0.010278048484716,0.720062691479206,0.693543821987009,0.0139196369938065,0.0112590882214452,
//   5.40303998757138e-8,2.94863096871154e-8,2.8095810140432e-11,-1.75592329471962e-11,
//   2.76919021734031e-10,6.17347666835937e-10,0.0139160218336218,0.0162277436182372,0.706674620057376,
//   0.706674620057376,0.0209959679182661,0.018003765798552,1.28517310158374e-7,7.91198676818941e-8,
//   9.50635341736372e-11,5.95467437713429e-11,-4.96910586938434e-11,-2.1722994258065e-11,
//   0.0200814001037142,0.0229512529972708,0.706174684169237,0.706174684169237,0.0310724964042737,
//   0.0271864624975419,4.13418290387606e-7,2.73480466035253e-7,7.10944779371368e-10,
//   3.71149327770687e-10,2.18087979651604e-10,1.87404002433835e-10,0.0275104098016373,
//   0.0308161379467153,0.705243901476086,0.705243901476086,0.0444819852991792,0.0397098449428893,
//   2.03895957399834e-6,1.44113320060014e-6,6.4601565762904e-9,4.01680399618611e-9,
//   1.38837431481968e-11,1.52441341773245e-11});
	// lower and upper limits for x
	Dvector xl(nx), xu(nx);
	for(int i = 0; i < nx; i++)
	{	xl[i] = -2.0;
		xu[i] = 2.0;
	}
	// lower and upper limits for g
	Dvector gl(ng), gu(ng);
    for(int i = 0; i < ng; i++)
    {
        gl[i] = 1.0;
        gu[i] = 1.0;
    }

	// object that computes objective and constraints
	FG_eval fg_eval;

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
	options += "Integer print_level  0\n"; 
	options += "String  sb           yes\n";
	// maximum number of iterations
	options += "Integer max_iter     100\n";
	// approximate accuracy in first order necessary conditions;
	// see Mathematical Programming, Volume 106, Number 1, 
	// Pages 25-57, Equation (6)
	options += "Numeric tol          1e-6\n";
	// derivative testing
//	options += "String  derivative_test            second-order\n";
	// maximum amount of random pertubation; e.g., 
	// when evaluation finite diff
	options += "Numeric point_perturbation_radius  0.\n";
    
//    options += "String hessian_approximation limited-memory\n";
//    options += "String hessian_approximation exact\n";
    
    options += "String linear_solver ma97\n";

	// place to return solution
	CppAD::ipopt::solve_result<Dvector> solution;

	// solve the problem
	CppAD::ipopt::solve<Dvector, FG_eval>(
		options, xi, xl, xu, gl, gu, fg_eval, solution
	);
    
    for(int i = 0; i < 2*dim; i++)
    {
        cout << solution.x[i] << " ";
    }
    cout << endl;
//    cout << solution.x << endl;
    cout << solution.obj_value << endl;
    cout << solution.status << endl;
    for(int i = 0; i < L; i++) {
//        cout << norm(i, solution.x) << endl;
    }
    
    std::vector<double> norms(L);
    for(int i = 0; i < L; i++) {
        norms[i] = abs(i, solution.x);
    }
//    std::vector<CppAD::AD<double>> xnorm(2*L*dim);
    ADvector xnorm(2*L*dim);
    for(int i = 0; i < L; i++) {
        for(int n = 0; n <= nmax; n++) {
        xnorm[2 * (i * dim + n)] = solution.x[2 * (i * dim + n)] / norms[i];
        xnorm[2 * (i * dim + n) + 1] = solution.x[2 * (i * dim + n) + 1] / norms[i];
        }
    }
    ADvector fg(L);
    fg_eval(fg, xnorm);
    cout << fg[0] << endl;
    for(int i = 0; i < L; i++) {
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

