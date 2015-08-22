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
T energy(/*int i, int n,*/ CppAD::vector<T>& fin) {//, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {

    //    vector<double> J(L, 0.2);
    double U0 = 1;
    //    vector<double> dU(L, 0);//.01);
    //    double mu = 0.5;
    double theta = 0;

    //    vector<double> dU({-0.062495818684028,-0.240152540894773,-0.245433857301291,-0.273625816197061,-0.242270144341017,-0.27259078540955,-0.20204475712817,-0.0848398286345957,0.424865322680541,-0.0971718211590916,-0.129687170384589,-0.271703531802801,0.0948468516335159,-0.117917878473979,-0.0135564488854456,0.419054844730518,-0.108478764896881,-0.216038019921792,0.260931974004009,-0.202439024332201,0.187098382438594,-0.183422731248427,-0.146060037885631,-0.156016398905386,-0.183701971564288});
    //    vector<double> J({0.138152396495495,0.140219774273089,0.140581397909088,0.140546956950688,0.140535853601678,0.140093899853519,0.138001669064538,0.129914946537351,0.13004950249811,0.137341045165717,0.139269569050823,0.136554490078846,0.134896150878805,0.136228606430583,0.129211199655538,0.130262768487484,0.138427531129939,0.133745485153161,0.13360220926095,0.134615642880126,0.134412331368716,0.138498071670926,0.138193037947643,0.138613245204837,0.137539881531959});
    //    double mu = 0.26669999999999999263;

    vector<double> dU3({0.043199239962729896, 0.08259186491718995, 0.19397213987527007, 0.31412171808007994,
        0.0915189876615099, 0.32699155669771995, 0.03970221015499997, 0.3169288596880999,
        -0.06649698521376701, 0.11030126264836992, 0.13055555283211007, -0.10016340121284895,
        -0.054463258625564004, -0.17320924136983595, 0.35891542195234005, -0.364378648638469,
        0.4271911797647201, -0.19375941242682504, -0.100964667952118, -0.019592374277792035, 0.26492282856173,
        0.28402138986372005, 0.025510001447129893, -0.017569415902372043, 0.060356175879900054,
        -0.09299207573504398, 0.39096937992084, 0.30643104688444, -0.353520812761554, -0.139929321387956,
        -0.34123939491588895, 0.13262668956456003, -0.391729203944317, -0.113887642455067,
        0.30318444229128994, 0.42098839070455996, 0.2522801656999001, -0.30002135151780596, 0.33836311285219,
        0.33843394927349, 0.44146187686278004, -0.023139212960445965, 0.27186045802215997,
        0.27349689504697006, -0.03372008677117999, 0.01805996653825992, 0.25448881206848006,
        0.1606392258635101, -0.31722957528124096, 0.19898802140462002});
    vector<double> J3({0.0199872024688441, 0.0214962626425955, 0.0238803449658659, 0.0227504318202043,
        0.0228723747345574, 0.0222827981052472, 0.0221899440052608, 0.0209501537515912, 0.0190945600203537,
        0.021156522094292, 0.0189093575813226, 0.0171830354126276, 0.0164309269687668, 0.0199880266581662,
        0.0174175184250734, 0.0178953155275678, 0.0202658069422593, 0.0157860325229642, 0.017509454115344,
        0.021035343501827, 0.0243318084090072, 0.0217241042749188, 0.0188015915058845, 0.0191413581434165,
        0.0183451209145919, 0.0212626001654235, 0.0258778593153581, 0.017195760116303, 0.0137369700547507,
        0.0138722624661954, 0.0160682496382933, 0.0154155045908814, 0.0135204219040843, 0.020262193793736,
        0.0261493969212978, 0.0255861770198534, 0.0175138263853036, 0.018162793198886, 0.0256728733632031,
        0.0267461529034936, 0.0225589323559612, 0.0210575213835723, 0.0242950606632568, 0.0209503493374639,
        0.0185636023452844, 0.0213662921171238, 0.0229271322131208, 0.016588609400904, 0.0168834475170639,
        0.0211207531674717});
    double mu = 0.9;
    vector<double> dU2({0.043199239962729896, 0.08259186491718995, 0.19397213987527007, 0.31412171808007994,
        0.0915189876615099, 0.32699155669771995, 0.03970221015499997, 0.3169288596880999, -0.06649698521376701,
        0.11030126264836992, 0.13055555283211007, -0.10016340121284895, -0.054463258625564004,
        -0.17320924136983595, 0.35891542195234005, -0.364378648638469, 0.4271911797647201,
        -0.19375941242682504, -0.100964667952118, -0.019592374277792035, 0.26492282856173, 0.28402138986372005,
        0.025510001447129893, -0.017569415902372043, 0.060356175879900054});
    vector<double> J2({0.0199872024688441, 0.0214962626425955, 0.0238803449658659, 0.0227504318202043, 0.0228723747345574,
        0.0222827981052472, 0.0221899440052608, 0.0209501537515912, 0.0190945600203537, 0.021156522094292,
        0.0189093575813226, 0.0171830354126276, 0.0164309269687668, 0.0199880266581662, 0.0174175184250734,
        0.0178953155275678, 0.0202658069422593, 0.0157860325229642, 0.017509454115344, 0.021035343501827,
        0.0243318084090072, 0.0217241042749188, 0.0188015915058845, 0.0191413581434165, 0.0197656098417165});
        vector<double> dU({0.01668792497172711,0.030749838262424722,0.06498445124364749,0.09367904602619714,
   0.03378884906859492,0.09629213501316491,0.01538717384900945,0.09425625126114379,
   -0.028438873575845736,0.0400093006628941,0.046458645241012286,-0.04418467682278704,
   -0.023035330483524352,-0.08169184627002968,0.10241305089083075,-0.2044184551207423,
   0.11384958881856644,-0.09311318677591873,-0.04457092676572827,-0.00802371952226777,
   0.08289075385715727,0.08723157134548964,0.010018492024330916,-0.007181777904576414,
   0.022944647709261634,-0.04075180447279647,0.10805624290869087,0.09207682057108912,
   -0.19638985929611386,-0.06401852960229315,-0.1874733203322776,0.04710325338256238,
   -0.22526114345790393,-0.05087570969364774,0.09139121724416088,0.11290042865214756,
   0.07990796558722835,-0.1587916857831292,0.09853089714094598,0.09854463872377184,
   0.11596723673667286,-0.009507449069859342,0.08449037933876347,0.08486388830129199,
   -0.013991262027431728,0.0071420305050362565,0.0804354009090511,0.0555558258794997,
   -0.1705366620644534,0.06634557679935082});
   vector<double> J({0.02723768386308467,0.028652550743929146,0.030777431433313254,0.02976420514670492,
   0.02986873726600082,0.029322829131598133,0.0292426681782343,0.028051489662438155,
   0.02634439851961236,0.028346004195243003,0.026144348999782737,0.0244198793795848,
   0.023613008173564994,0.02703939931627623,0.02427653764661395,0.02470233164031898,
   0.027237236423487663,0.022918777947536148,0.024756692054584112,0.02817697372438214,
   0.031173310416124176,0.028824887251795318,0.02607503310364909,0.026409230050398493,
   0.025600095187419854,0.028284489315528467,0.03245801925566863,0.024106085935874485,
   0.020570169820478188,0.020732372829626377,0.02305487041454044,0.02229661580647715,
   0.020288705767379515,0.027379758528124184,0.03267478189590423,0.03220579980669879,
   0.02453567299736222,0.02513262444294944,0.03229462027675042,0.03315656399160753,
   0.029483306028597115,0.02819352492431643,0.03114223812758261,0.028088011601379075,
   0.02583610155818003,0.028502466130532502,0.02994812674963089,0.02361443334808476,
   0.023900306518797355,0.02829607624426654});

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

typedef CppAD::AD<complex<double>> ADc;

using namespace CppAD;

class FG_eval {
public:
    typedef CppAD::vector<CppAD::AD<double>> ADvector;

    void operator()(ADvector& fg, const ADvector& x) {
        ptime begin = microsec_clock::local_time();
        fg[0] = energy(const_cast<ADvector&> (x));
        cout << fg[0] << endl;
        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;
        for (int i = 0; i < L; i++) {
            //                fg[i+1] = norm(i, const_cast<ADvector&>(x));
        }
    }
};

class thread_id {
public:
    int id;

    thread_id(int i) : id(i) {
    }
};

boost::thread_specific_ptr<thread_id> tls;

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

    // object that computes objective and constraints
    FG_eval fg_eval;

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
    options += "Numeric tol          1e-14\n";
    options += "Numeric acceptable_tol          1e-14\n";
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
            options, xi,
            xl, xu, gl, gu, fg_eval, solution
            );

    cout << solution.obj_value << endl;
}

bool parallel = false;

bool in_parallel() {
    return parallel;
}

size_t thread_num() {
    return tls->id;
}

double energyfunc(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    CppAD::ADFun<double>* func = static_cast<CppAD::ADFun<double>*>(data);
    CppAD::vector<double> cppx(x.size());
    for (int i = 0; i < x.size(); i++) {
        cppx[i] = x[i];
    }
    CppAD::vector<double> cppgrad = func->Jacobian(cppx);
    copy(cppgrad.data(), cppgrad.data()+grad.size(), grad.begin());
    cout << func->Forward(0, cppx)[0] << endl;
    return func->Forward(0, cppx)[0];
}

void thread_func2(int i) {
    tls.reset(new thread_id(i));
        ptime begin, end;

    CppAD::vector<CppAD::AD<double>> qw(2*L*dim);
    for(int i = 0; i < 2*L*dim; i++) {
        qw[i] = 1;
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
        cout << "E0(" << i << ") = " << E0 << endl;
}

/*
 * 
 */
int main(int argc, char** argv) {

    cout << setprecision(16);
    
    /*std::vector<double> fin(2*L*dim, 0);
    for(int i = 0; i < L; i++) {
        fin[2*i*dim+2] = 1;
    }
    std::vector<double> fin2({1.0742395016777295e-9,-3.137268727051826e-9,0.7071067811865475,0.7071067811865475,
   -1.8457963265314524e-9,3.0632256736259465e-9,-1.1484846404075058e-10,7.963440315043461e-11,
   6.459148681639323e-14,1.1091372035300614e-12,-7.898684818559355e-11,-4.966520185186698e-12,
   8.979891680048182e-9,1.6911075303423993e-9,0.7071067811865475,0.7071067811865475,
   3.880985987242594e-9,9.399235422253854e-9,-7.898236897572482e-12,-5.4575607940612385e-11,
   2.39995434425278e-13,7.331816751889177e-13,-3.604935799980046e-11,-2.4179616254904507e-12,
   1.0308646340160355e-8,-7.28316885387006e-10,0.7071067811865474,0.7071067811865474,
   1.8036815010573194e-9,1.232091356044776e-8,1.3778238924517976e-10,-7.538657980025289e-11,
   -8.607899839342414e-12,-1.5729600027775903e-12,7.173242422631578e-12,7.324018790824572e-12,
   1.2318859333523456e-8,6.385014592641985e-9,0.7071067811865474,0.7071067811865474,
   -5.599440640343882e-9,6.837193824039673e-9,-2.4840599673316085e-10,-1.4068274231012572e-9,
   -1.1364890884738477e-12,-3.086131326963239e-13,-4.2092860012494474e-14,-2.2815410975142493e-13,
   3.302394048563186e-9,8.756659807657615e-10,0.7071067811865475,0.7071067811865475,
   1.4189448306843413e-9,7.551238698727646e-10,-3.1092798611522403e-10,5.562123104876829e-10,
   4.314164775481483e-13,4.3191377139405095e-13,-4.3593862719328574e-12,-1.0824931266418638e-12,
   1.1054227505343672e-8,1.0380280562582895e-8,0.7071067811865472,0.7071067811865472,
   -1.1465485368213427e-8,6.843427071008177e-9,-2.399193829992065e-10,-1.154398656244163e-9,
   -6.6008138090326e-14,-9.149757543780295e-13,-3.607905384069352e-14,-1.0936869312251786e-13,
   1.244122321946795e-8,7.354806796012133e-9,0.9785206800706363,0.20614868099044278,
   -6.54673415776475e-9,-6.7547175117244944e-9,5.080573424666717e-11,6.472875673845821e-11,
   -4.675174841284009e-10,-5.29902332488503e-10,-6.702328812570969e-12,-2.6066886363644684e-10,
   1.5703338398307486e-8,1.758505836504679e-8,0.7071067811865472,0.7071067811865472,
   -2.127369285682463e-9,-1.6287593590274905e-9,1.6980798105353352e-10,1.334136639935082e-10,
   -1.625225410177399e-13,-9.194952344938543e-13,-3.012736504432407e-13,-1.144526968930243e-13,
   -2.226458963622365e-9,1.0014493061527111e-8,0.18711847462358808,0.9823373536899336,
   -1.3895004415738072e-9,-2.018240355326381e-9,-3.805852138106876e-12,-1.3069587723566562e-11,
   1.1263091677372054e-10,-9.230120075175823e-12,-6.229163180915956e-10,-7.386959697747759e-10,
   7.73552238012678e-9,1.5750351301616678e-8,0.8001028825213883,0.5998627987973294,
   -1.039247569313165e-9,-4.2713927960834794e-9,4.726165795573911e-12,3.749615732759786e-12,
   -2.475313551238027e-14,-4.114014694535888e-14,-1.0112041545772653e-10,1.6635239378438612e-10,
   -3.2370854406179788e-9,1.7130080679656723e-8,0.7071067811865474,0.7071067811865474,
   2.9277542477344343e-9,-2.362076215918223e-9,-2.2175135847750463e-11,-4.929109096228734e-11,
   1.0790650915588979e-12,7.966251470599762e-13,-2.8278441219386186e-11,-1.4463487818639925e-11,
   -1.2798369566066529e-8,8.743351535138375e-9,0.7071067811865474,0.7071067811865474,
   -1.6738469261732887e-9,-7.796352767528075e-9,-1.3315771648169878e-11,-3.4372364207991943e-12,
   7.651853805011535e-14,9.034167205757711e-14,-1.7413243896663982e-10,-1.0336681937367465e-10,
   -7.934017613650271e-9,-5.6663068227916295e-9,0.7071067811865475,0.7071067811865475,
   -4.61830742368923e-9,-7.0968174699154746e-9,-2.0072499710650646e-12,-1.0075445654748728e-11,
   -2.77083273311049e-12,-8.678138037043871e-13,4.4209242439595014e-11,-2.073194332407385e-12,
   1.000793638468538e-9,-1.1418571513394326e-9,0.24834242694007078,0.9686723073266396,
   -1.6428599751231497e-9,-3.861015438813049e-9,-5.116554265170751e-11,1.7978490185183009e-12,
   -3.35374547757071e-10,4.454028061209648e-11,6.7065079700884e-10,-4.228422995433422e-10,
   -1.4194508627112988e-8,-1.9827069638577614e-8,0.7071067811865471,0.7071067811865471,
   -1.1479168434775084e-8,-6.980362127621799e-9,7.926819247390347e-10,-4.970635686633579e-10,
   -3.3107636679797234e-12,-2.1306767965935415e-12,-1.6726990162522668e-13,-1.917505098961457e-13,
   -2.1973811702551194e-8,-2.880127628273244e-8,0.774986716232096,0.6319775230684959,
   3.866295283138864e-9,3.849715416311502e-9,-2.8029567606406383e-11,3.5742163616090756e-11,
   -2.0845106923031873e-10,-1.6105923498874966e-10,2.674698892667013e-11,1.960378685199562e-10,
   -2.275747786337695e-8,-5.3910852498498204e-8,0.7071067811865462,0.7071067811865462,
   -2.9563700945532115e-10,1.3747540224363382e-9,-1.1098142055646589e-10,-1.7647038458344347e-10,
   -9.505191941576114e-13,-2.059090424675924e-12,-2.074137445671611e-14,-1.3485132481897957e-14,
   -2.4148055590701388e-8,-5.1176577648146034e-8,0.7071067811865464,0.7071067811865464,
   5.489908515539089e-9,3.849528248094541e-9,-2.253032937246288e-13,6.8348252496370155e-12,
   1.7209186988339018e-14,1.3235118083213784e-13,3.055003948476179e-10,-2.7148741169428955e-10,
   -9.04729120280802e-9,-2.67089624518009e-8,0.7071067811865471,0.7071067811865471,
   -5.833674697177632e-10,-1.237118250037861e-9,4.5651690016438606e-12,1.7501087019416564e-12,
   2.653190459952007e-13,2.896005153872076e-13,-9.628323014812197e-11,-2.4145708188256936e-11,
   -7.377287822186021e-9,-2.0014607604604314e-8,0.7071067811865472,0.7071067811865472,
   -7.57582240257844e-9,-2.488355726134783e-9,-1.0564454347121011e-10,-3.325947578664214e-11,
   1.8087268687762921e-13,5.927784953038581e-13,-3.639898930206073e-11,-6.204199578778214e-11,
   -9.354728697307202e-9,-1.768171333447541e-8,0.7071067811865471,0.7071067811865471,
   -1.3131175952672928e-8,-1.2850110037063526e-8,4.731091760465498e-10,-5.799578967155441e-11,
   5.147933431276034e-13,1.14398098696373e-13,-1.763769275153374e-13,-4.248733928503832e-12,
   -1.0476943652915793e-8,-1.018577748627643e-8,0.7071067811865474,0.7071067811865474,
   -5.269785718036448e-9,-1.3232373628575553e-9,-1.6766135230175825e-10,1.4961352526965088e-10,
   -2.100393869423288e-14,-4.1833640481985354e-15,5.062532947086988e-15,-2.0716810034420282e-14,
   -7.377757552413831e-9,2.9926847820291045e-9,0.7071067811865475,0.7071067811865475,
   -4.5108978044414e-11,-2.516533098669071e-9,-3.7992227838913383e-11,-2.435088156979515e-10,
   8.837924270329859e-13,4.2397114689788657e-13,-7.858177758507471e-11,-4.085340596270202e-11,
   -9.602938407554568e-9,2.0722211035871192e-9,0.7071067811865475,0.7071067811865475,
   -1.7192100755868238e-9,6.834545176345447e-10,1.297329264831244e-10,1.7361516353527966e-10,
   5.093928448288921e-14,1.4929232946278896e-13,-5.537821655333365e-11,-5.34967090084705e-11,
   -1.530631426962912e-8,6.576001534451081e-9,0.7071067811865474,0.7071067811865474,
   5.857775993010143e-9,-4.129986229828198e-9,-2.6069073354353106e-10,-1.2356610326243535e-10,
   3.4399461932280164e-13,5.661128306640314e-13,-7.43013071157464e-11,-4.727565157261873e-11,
   -1.0401075446519226e-8,2.0575082333318625e-8,0.7071067811865472,0.7071067811865472,
   3.5993841915597267e-9,1.6766536409801398e-9,2.634173891819326e-11,-9.609600282555775e-12,
   5.741258320506009e-13,2.0437338971331437e-13,1.3704082326425051e-10,1.047027830025623e-10,
   -1.2662592644084619e-8,8.78441699102033e-9,0.7071067811865472,0.7071067811865472,
   2.2766217115023438e-8,2.978311482165143e-10,1.7061177571689109e-10,1.3010617000946758e-11,
   -3.743191198888168e-12,-2.228711475805845e-12,-2.912144506105875e-13,-4.18245939206844e-14,
   -9.732607881502258e-10,1.899973545282552e-8,0.3638999198298004,0.9314380539509131,
   1.4183367065171534e-8,1.632875043095772e-8,-2.326965368014191e-11,-9.775711844292428e-12,
   -1.097871288231946e-13,-3.3435698859814954e-13,9.159835445080201e-10,1.1638213556764698e-10,
   5.477237239178189e-9,1.9133744005086587e-8,0.7071067811865472,0.7071067811865472,
   1.3443127029450094e-9,-2.0039002245276205e-9,-2.83973073771871e-12,-4.147794006721308e-12,
   1.973245010251446e-12,3.939270430083901e-13,-1.8498807449625112e-10,-3.9124388206187336e-10,
   -9.133911544879426e-10,3.833684884657172e-9,0.7071067811865475,0.7071067811865475,
   2.272460520232667e-9,-6.329656457660006e-10,-3.223283789070989e-11,-8.085648938528021e-12,
   1.2553992323807377e-14,-7.347975192214533e-16,3.751408158691907e-11,6.196343243099029e-11,
   1.218696310033215e-8,2.5993897176405104e-9,0.7071067811865474,0.7071067811865474,
   -3.3899395225907027e-9,7.446670201475446e-10,-2.7593085041300706e-12,-1.0109614261343192e-11,
   -8.954284322990029e-13,-1.1310503806904282e-12,5.819864935296164e-11,-1.0180918332230821e-10,
   -3.778576473227595e-11,2.214132023597219e-8,0.7071067811865472,0.7071067811865472,
   1.3471964602108291e-9,6.102048656844384e-9,6.718905662378387e-11,8.380842933471888e-11,
   -7.486155164747789e-13,-1.4424168058190298e-12,1.989151515488784e-12,3.4617760867278576e-12,
   -1.3911837478850676e-8,2.380442015792719e-8,0.5142811008150238,0.85762168194635,
   6.323332342210936e-9,2.2028611174857833e-9,4.4410243380952455e-13,8.777195157404287e-13,
   -3.0173173075495963e-10,-1.2646434286334527e-10,-1.6452756118559067e-10,1.1079387581147136e-9,
   -3.306302134308372e-8,3.5072058332876864e-8,0.7071067811865465,0.7071067811865465,
   2.0202939727443793e-8,-1.7957510850949562e-8,-2.99010861425829e-11,-5.741604940801266e-11,
   1.080937795481666e-13,6.116438739221166e-14,-3.7457166178503806e-10,-1.2534589243285764e-10,
   -4.424198658955841e-8,5.363594913229376e-8,0.7355999265543607,0.6774162295466551,
   4.531296699443805e-8,-7.589525327148413e-8,9.634763997116756e-11,2.6806903256258347e-10,
   4.745431511190526e-13,9.467456007416727e-13,-6.638842560559486e-12,-4.965876812647577e-12,
   -7.185695775289205e-8,5.939320930631953e-8,0.7071067811865389,0.7071067811865389,
   6.368792550157736e-8,-1.0562293037561937e-7,-4.814840084677393e-11,-6.493148478118105e-11,
   -5.1625053508520244e-12,-2.712430159900363e-12,-2.890241373485395e-14,-3.194010521272787e-14,
   -8.67420363722141e-8,5.790490683275408e-8,0.7071067811865395,0.7071067811865395,
   6.996171036536259e-8,-8.206563106345796e-8,6.336069213745468e-11,-1.4895652495562005e-10,
   3.137010491381983e-13,2.938551570630223e-13,-2.2745975931631217e-13,-4.3060870454403296e-13,
   -1.2203216509861929e-7,8.683016440618261e-8,0.7071067811865381,0.7071067811865381,
   4.246221373795389e-8,-4.4537384762963756e-8,4.820306059754618e-11,8.101831465410626e-11,
   -1.0871647438409294e-12,-1.6969158928017665e-12,8.275272776956583e-11,-1.2743684220361611e-11,
   -1.2285759414504503e-7,9.634699323723385e-8,0.7071067811865251,0.7071067811865251,
   1.5153714330865775e-7,-1.255899574455082e-7,-1.0461622990493765e-9,-1.2441952253573255e-9,
   -1.4958270912736885e-12,-2.142520812345993e-11,-2.724772408903255e-12,5.8640848876277786e-12,
   -1.1525116952389724e-7,1.1847917452751016e-7,0.7071067811865175,0.7071067811865175,
   1.8602948143044568e-7,-1.5126791575936645e-7,2.341661378658909e-9,1.7281972056096552e-9,
   -7.257294589226447e-13,-1.055454938825509e-12,-2.2003803034562236e-13,-3.9933934499546523e-13,
   -9.627789421259894e-8,1.2157609613262726e-7,0.7071067811865163,0.7071067811865163,
   1.87210257310363e-7,-1.7082239675726238e-7,5.345058266494056e-10,6.296555711207905e-10,
   -2.900522779606517e-12,-2.5732965357078728e-12,8.527465998229706e-15,1.030569840964898e-14,
   -1.1299270250430153e-7,1.2075183814296573e-7,0.7071067811865313,0.7071067811865313,
   1.0409928594097877e-7,-8.73829757478646e-8,-3.1942568903627734e-10,-1.030971952923352e-10,
   1.1803388444958384e-12,1.8787097747172568e-13,-1.0185705831752572e-10,-6.677235175074416e-11,
   -8.52188744428214e-8,7.129497230288253e-8,0.7071067811865374,0.7071067811865374,
   9.696254199214865e-8,-8.24782188114827e-8,7.252391558562752e-10,-3.904157424177939e-10,
   8.158041953704995e-14,2.045365550675298e-13,-9.299445593193055e-13,-7.649822262570343e-13,
   -6.702406369031811e-8,4.250813815781333e-8,0.7071067811865421,0.7071067811865421,
   6.267679901300884e-8,-7.00606938679497e-8,-2.2652436140203923e-10,6.110788716109469e-10,
   5.544050385889326e-14,-1.4540386119255776e-14,-1.0630888086791444e-13,-2.244675838074914e-13,
   -4.6859642814635316e-8,5.343028791209415e-8,0.7071067811865451,0.7071067811865451,
   2.1899406753574303e-8,-3.3563492106654635e-8,1.4136296262493318e-11,-4.662974430475595e-11,
   2.104745847582519e-13,1.1349730960785477e-13,-9.424543023694799e-12,-7.572582234007293e-13,
   -2.7674572229261637e-8,4.1262921400037046e-8,0.7071067811865465,0.7071067811865465,
   1.76163540483446e-8,-1.500195789850849e-8,-7.717112990497138e-12,-1.3048445904495312e-10,
   1.8976829242368378e-13,1.0895064669019711e-12,-4.630809505853604e-11,-1.5880186527910107e-12,
   -9.456037071514407e-9,4.3011587014558126e-8,0.7071067811865466,0.7071067811865466,
   1.0092144244688191e-8,-1.7414050154953963e-8,7.557083499213176e-11,2.383054263530253e-10,
   2.4324573491236035e-13,6.916320302658513e-13,-1.3393248061753479e-12,-1.0929232635091036e-12,
   -1.2079572901186249e-8,1.1112093238904699e-8,0.5796275247445533,0.8148815451085526,
   1.0805259197638464e-9,-1.1675424283003963e-8,-1.062646552458279e-11,-4.496455558294696e-11,
   5.440630562327005e-17,2.252886142535261e-16,1.4106750102422666e-9,-1.4243107695634692e-9,
   -1.6819489211752927e-8,9.17358881848518e-9,0.7071067811865472,0.7071067811865472,
   -6.762911803886738e-9,-1.1292265482273675e-8,-1.8714435655531136e-11,-1.5560010447359836e-12,
   3.849746217693516e-13,-1.7277496740976536e-12,4.11995701236231e-11,4.3615753215851635e-11,
   -2.9250873437370238e-9,-6.794755951607345e-10,0.7071067811865475,0.7071067811865475,
   3.623018449459509e-9,1.4298773144431045e-9,1.8140316845931425e-10,4.658208584769844e-13,
   7.380139129488344e-13,7.543467338377316e-13,-1.372911995025352e-11,-8.973249788433026e-12});
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
    CppAD::CheckSimpleVector<size_t, CppAD::vector<size_t>>();
    CppAD::CheckSimpleVector<set<size_t>, CppAD::vector<set<size_t>>>(CppAD::one_element_std_set<size_t>(), CppAD::two_element_std_set<size_t>());
    parallel = true;
//    CppAD::RevSparseJacSet()
    
    thread_group threads;
    for (int i = 0; i < 1; i++) {
        threads.create_thread(boost::bind(thread_func, i+1));
    }
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
    Dvector xiii({-2.97732436435043e-7, -3.39536984289482e-7, 0.707106779793288, 0.707106779793288,
        0.0000447988522968909, 0.0000439726284618764, -5.71206755151725e-10, -2.09085281501439e-9,
        -6.49068351796545e-14, -2.14080285277463e-13, 1.37155365629369e-9, 2.57359698636968e-10,
        -8.35098567863845e-8, -9.48944036628442e-8, 0.70710678111192, 0.70710678111192, 0.000010504669330809,
        0.0000100356954370684, 4.22738899136155e-9, 5.49175209611633e-10, 8.55798756027178e-14,
        3.44580775429171e-13, -2.13161193718045e-10, 2.39282598204761e-10, -8.73765300801373e-8,
        -7.11081595524319e-8, 0.741055179072384, 0.671444131377879, 2.19537399158575e-6, 1.83595254569312e-6,
        -1.85626191285956e-10, -1.32277302605548e-10, -1.5913677551844e-11, 2.57176527000595e-10,
        2.51540221351124e-10, 1.10943483867843e-10, -1.44567034772317e-7, -1.61734467671088e-7,
        0.707106781180081, 0.707106781180081, 3.02797048792772e-6, 3.01237631605123e-6, -2.8068741383869e-11,
        -1.16879928899916e-11, 2.18785394991075e-10, -2.59172039975633e-9, 1.86081559696853e-10,
        3.93839461427624e-11, -1.18765102091146e-6, -1.18899628918504e-6, 0.707106780793875, 0.707106780793875,
        0.000023650303833997, 0.0000234197422285229, 1.47226454598038e-9, 7.577704516697e-10,
        1.56960989479038e-13, 2.93812110214782e-14, -6.03603626210946e-12, 3.4914862761243e-10,
        -2.59158416844655e-6, -2.60946033249112e-6, 0.707106775671369, 0.707106775671369,
        0.0000886094625024301, 0.0000879438334615934, -1.58536347231633e-9, -1.51359666240925e-9,
        6.02592684215659e-10, -2.38441983533862e-9, -4.01159997927335e-11, 5.06009550957416e-11,
        -0.0000231689353287103, -0.0000233503813229497, 0.707106324809083, 0.707106324809083,
        0.000806285006014607, 0.000799782738809736, 3.61361525190211e-8, 2.56226854673184e-8,
        6.71439105889468e-13, 4.46561779793067e-13, -8.38320782432674e-11, -2.15027556615421e-10,
        0.0000410675928185923, 0.0000414063506823284, 0.707103215494357, 0.707103215494357, 0.0022542631216858,
        0.00223610581030883, -1.1563058961876e-6, -1.13767392169533e-6, -7.5542212444702e-9,
        7.7796216056013e-9, -3.34016864016247e-10, -1.53608481667394e-10, -0.0000878471862291269,
        -0.0000885829444738739, 0.706780877676829, 0.706780877676829, 0.0215528907892829, 0.0213784737972014,
        6.88921523829568e-6, 6.7780006743433e-6, 6.95663661864654e-10, 6.78003164556414e-10,
        1.06719664741958e-9, 5.99386239357794e-12, -0.000167511488231103, -0.000168868482036004,
        0.706909317583052, 0.706909317583052, 0.0167765526960232, 0.0166410350091714, -6.58478463238096e-6,
        -6.4782141772705e-6, -4.79915950232947e-10, -4.70033074337385e-10, 1.38104963941297e-10,
        3.14243985050174e-11, 0.000994700903463984, 0.00100281196881454, 0.704433004790406, 0.704433004790406,
        0.06167393296153, 0.0611751819359579, -0.000327875953340657, -0.000322594345289331,
        5.50979418765813e-7, 5.37720185636812e-7, -3.75640930344145e-10, -1.80641268471649e-10,
        0.000102465079318333, 0.000177922612233314, 0.358887326850546, 0.617424522880897, 0.354221838628228,
        0.603743510287931, 0.0013922266380168, 0.0023510692581972, 2.18816654826113e-6, 3.66123445125511e-6,
        1.6843357179485e-9, 2.79285693766985e-9, 0.00136809897642752, 0.00137926970350548, 0.645734931812886,
        0.645734931812886, 0.289305270119381, 0.286969469971523, 0.0000532372200110433, 0.0000523889345256495,
        2.29710496224685e-7, 2.24190583131434e-7, 9.23082906440196e-11, 7.43840523801048e-11,
        0.0000478289055868938, 0.0000486133551421711, 0.122004411520455, 0.122994763889637, 0.696399944420688,
        0.696399944420688, 0.00457030305581167, 0.00453364717097104, 9.27838273051841e-6, 9.13022647408829e-6,
        9.44523925242649e-9, 9.23385520199532e-9, 0.00078441753357453, 0.000790144106616635, 0.706996677479575,
        0.706996677479575, 0.012500466942947, 0.0124052890405478, -0.000042504120484814, -0.000041839297309247,
        7.38469958635832e-8, 5.15834369894024e-8, 3.33758392317603e-11, 1.15943883803258e-11,
        -6.00917678941098e-7, -5.97346745501991e-7, 0.00361733603775321, 0.00361786749258695, 0.70709189904631,
        0.70709189904631, 0.0028211870397, 0.00282124251802116, 2.69965830075878e-6, 2.70113701826759e-6,
        1.03966622936577e-9, 1.04061068849448e-9, 0.000875820720248044, 0.000869095333566123,
        0.707019900227435, 0.707019900227435, 0.0110052689811856, 0.0110941316261711, -0.0000401245385263177,
        -0.0000407954162615835, 7.0803290703911e-8, 6.9847804855014e-8, 1.37251972042273e-10,
        -2.12191206871689e-10, 0.000117711397569582, 0.000115688049243522, 0.118268534951297,
        0.117265663910159, 0.697204594384136, 0.697204594384136, 0.006007449985866, 0.00605868213427749,
        0.0000160369954476481, 0.0000163115738193909, 2.23749847969886e-8, 2.29715056031948e-8,
        0.000447918156076484, 0.000440296885230161, 0.468683385951292, 0.464700475523993, 0.531218392795365,
        0.531218392795365, 0.00121862432736905, 0.00122904531086879, 1.249265477592e-6, 1.27110385675849e-6,
        7.96046852005004e-10, 8.16344681797925e-10, 0.000813173246362252, 0.000806267780931596,
        0.690411495453591, 0.690411495453591, 0.152093285555145, 0.153395653468522, -0.000118473933338422,
        -0.000120509489648196, 4.44727955826266e-8, 4.56251199321782e-8, -6.82548714367525e-10,
        -2.11862159936119e-9, 0.000111730374072348, 0.000110780687523443, 0.706883321556297, 0.706883321556297,
        0.0176991712476258, 0.0178507327457936, -0.0000511941939359365, -0.000052074658382139,
        5.15921722674104e-8, 5.28415468218596e-8, 1.6680539268788e-10, 1.10335698764651e-10,
        -0.000175022209851087, -0.000173582912935224, 0.707102226474984, 0.707102226474984,
        0.00252123800031406, 0.00254268029240167, -3.94645199461497e-7, -4.01326977928504e-7,
        7.41902365613315e-11, 3.84328032403537e-11, -2.1048209670296e-10, -6.05782006292904e-11,
        -0.0000343718228744168, -0.0000340691024919998, 0.70710593061927, 0.70710593061927,
        0.00109223275506415, 0.00110020525074994, 3.44715383850087e-8, 2.05369497221589e-8,
        3.95772011819479e-13, 2.77975725885936e-13, 1.24439309713967e-9, -8.67546588268484e-10,
        -2.57430086797579e-6, -2.56824803132064e-6, 0.707106553684216, 0.707106553684216, 0.000565966098878404,
        0.000568456518743242, 1.45672581959269e-9, -2.62703539695158e-9, -1.22632666204335e-14,
        -4.41328557461623e-14, 8.07318338340527e-11, 2.84143403769538e-10, 1.42668155019231e-7,
        1.50205628201075e-7, 0.707106765876955, 0.707106765876955, 0.000147264417312083, 0.000147021136892483,
        -1.51316333240863e-10, -8.93197498649441e-10, 1.52522819413177e-13, 2.54868455676061e-13,
        3.40021698984918e-10, -6.48994618983208e-10});
    Dvector xii({-6.79045819393098e-9, 7.71152338744772e-9, 0.707106781186545, 0.707106781186545,
        -6.63442720769046e-8, -3.48016745636222e-8, -2.45211700986609e-8, 1.9287403708982e-10,
        2.49691469412262e-13, 6.11976813244223e-14, 1.85148291581534e-9, 1.45525846918271e-9,
        -2.19595679203122e-8, -7.19724930329616e-8, 0.672114580890689, 0.740447155544597, 1.5995539027605e-7,
        1.31613650266916e-7, 9.86793472857681e-10, 7.69215080101853e-10, -2.63713289890294e-12,
        -2.40718213834916e-12, 2.32618006979149e-9, 3.4377638387667e-9, -9.12212908614002e-8,
        -3.62607218609557e-8, 0.707106781186394, 0.707106781186394, 4.55861667288651e-7, 4.64834769177162e-7,
        -1.13156629207678e-10, -3.78869850132558e-10, 2.08588895649252e-12, 1.53657718041163e-11,
        -8.20431050243829e-10, 4.70700553372067e-10, -1.47442058476654e-7, -1.46850147874533e-7,
        0.707106781181074, 0.707106781181074, 2.74749886940537e-6, 2.80871028900977e-6, 4.05678469217281e-12,
        4.12310879285748e-11, 1.2512166032571e-10, 6.84411346908066e-11, -5.63594708611019e-10,
        -9.79463701480509e-10, -1.23981924758914e-6, -1.17876290597637e-6, 0.707106780790895,
        0.707106780790895, 0.0000235946392532646, 0.0000236525241888609, -3.3032723392341e-9,
        -7.08778218797757e-9, -2.29179615900475e-13, -1.74254353723683e-13, 3.78078621110417e-9,
        -6.02715237838555e-9, -2.64072741758591e-6, -2.60229492348438e-6, 0.707106775689522, 0.707106775689522,
        0.0000879767965346132, 0.0000882851789562232, -1.54926795477873e-9, -1.53765339064787e-9,
        -2.77222904812182e-10, -2.64671352214544e-10, 5.08979294511079e-10, 1.55607081561332e-10,
        -0.0000233288115866426, -0.0000232196754262957, 0.707106324797759, 0.707106324797759,
        0.000801738462356886, 0.000804359410653844, 3.52880231908674e-8, 3.23093473791516e-8,
        5.17113020890157e-13, 4.9773532135885e-13, 1.84725788696705e-11, -2.8522550225063e-10,
        0.0000411672513764212, -0.0000412622655707445, 0.70710321554808, -0.70710321554808,
        0.00224875182543782, -0.00224161513422804, -1.15074795910339e-6, 1.14337468971707e-6,
        5.25595575278837e-9, -1.51869185689529e-9, 1.03378173496285e-10, -2.38357816203738e-11,
        -0.0000883728195115809, -0.0000880939478759592, 0.706780891988618, 0.706780891988618,
        0.0214314865260442, 0.0214992362910836, 6.81428699994571e-6, 6.85541183396645e-6, 6.85874101375741e-10,
        6.92552896373652e-10, -9.31130005827594e-10, -3.93550762801282e-9, -0.00016845107005252,
        -0.000167964867372416, 0.706909327262225, 0.706909327262225, 0.0166843269531305, 0.0167326811986034,
        -6.51482556120043e-6, -6.5544311215553e-6, -4.70835106728276e-10, -4.75221161169962e-10,
        3.60922539946126e-9, -2.8340488235108e-9, 0.00100019252842316, 0.000997386961863291, 0.704433028876251,
        0.704433028876251, 0.0613388638411128, 0.0615105901897412, -0.000324333917046785,
        -0.000326151381967918, 5.42123078063428e-7, 5.46660410661889e-7, 5.56652400848149e-9,
        -6.99958652906872e-9, 0.000195520724686646, 0.0000627627135507756, 0.679676735071237,
        0.219208572859753, 0.665893572758911, 0.215790127385673, 0.00259811573690974, 0.000845977455054579,
        4.05386366316449e-6, 1.32627331273447e-6, 3.08802342788574e-9, 1.01460948965434e-9,
        0.00137560974453084, 0.00137176938419611, 0.645734641079724, 0.645734641079724, 0.2877352934034,
        0.288544914902567, 0.000052669656317455, 0.0000529679273013292, 2.26012284415414e-7,
        2.27934338658326e-7, 3.76458683664844e-10, 1.02211004127463e-9, 0.0000483591646561038,
        0.0000480913910648718, 0.122677430441269, 0.122326128159832, 0.696399712598017, 0.696399712598017,
        0.00454549821924514, 0.00455857890652957, 9.17809760902477e-6, 9.23118704297534e-6,
        9.33412061497656e-9, 9.44108805608717e-9, 0.000788824814391602, 0.000785748037611256,
        0.706996673806915, 0.706996673806915, 0.0124318307049253, 0.0124744873769239, -0.0000420425498251253,
        -0.0000423057040938332, 6.25323360471957e-8, 6.48415289674283e-8, 1.47093932986539e-9,
        5.18853384845225e-10, -8.07153705424467e-7, -2.58023903249908e-7, 0.00486215638268478,
        0.00159153755210319, 0.948292402901197, 0.317331730706734, 0.00377496770816389, 0.00129125271257957,
        3.60570938565126e-6, 1.26035702348124e-6, 1.69910697208368e-9, 9.14049721900764e-10,
        0.000882558652323458, 0.000862260843329872, 0.707019900769935, 0.707019900769935, 0.0109180496440999,
        0.0111799080811817, -0.0000394781014815146, -0.0000414214765238932, 6.50100578739359e-8,
        7.93273352572673e-8, 2.28960450453013e-10, -2.36855900450405e-10, 0.000119547409277435,
        0.000113841557862962, 0.1191921098923, 0.116328960099172, 0.697204412689051, 0.697204412689051,
        0.00595947738354294, 0.00610611906606808, 0.000015777604426655, 0.0000165633864298963,
        2.18201229038419e-8, 2.34722815589608e-8, 0.000458133344518428, 0.000429543355577029,
        0.475965846068775, 0.457258257087749, 0.535378109158818, 0.527008811316418, 0.00121917823228314,
        0.00122960851014038, 1.24183586723109e-6, 1.28252338705774e-6, 7.85968401503615e-10,
        8.31252251861572e-10, 0.000816217197821242, 0.000797974993011171, 0.689834315430512, 0.690946323432342,
        0.150843608247971, 0.154811723955436, -0.000116150377269492, -0.000122069686615712,
        4.35396083857665e-8, 4.69116222924851e-8, -2.79516917403873e-9, -3.06783725549174e-9,
        0.000096527246470433, 0.0000945966928410121, 0.706878057559358, 0.706878057559358, 0.0177606392450249,
        0.0182032811513461, -0.0000509783340179075, -0.0000535379465109295, 5.08042090348981e-8,
        5.46348258181621e-8, 9.68048970811921e-10, -4.30035874916469e-10, -0.00022123047634171,
        -0.000143468507478423, 0.831847836881644, 0.55497604640512, 0.00455783277309131, 0.00314973373542979,
        -1.52809234287719e-6, -1.09951456738585e-6, -1.02700542009539e-8, -2.99738735290722e-9,
        2.36520143952414e-10, 2.69390935050993e-10, -0.0000341539768836597, -0.0000333443380107547,
        0.706989630909508, 0.706989630909508, 0.0125717177382487, 0.0131633280171592, -1.193322671445e-6,
        -1.31714600605584e-6, 5.10467046727844e-11, 5.87271280581108e-11, -7.30415001847096e-11,
        -2.57937195081837e-10, -0.000103380003614199, -0.0000985957559859386, 0.70607078610147,
        0.70607078610147, 0.0373407353461938, 0.0391629739956144, 0.0000160775042222112, 0.0000176831790492965,
        2.89856656148333e-9, 3.34376370005585e-9, -4.4253510519603e-9, -1.86917690241013e-9,
        0.000738693188481874, 0.000704103277040555, 0.703686605032618, 0.703686605032618, 0.067775860702201,
        0.0711030145373026, -0.000185879509026512, -0.000204560108842726, 1.26768766460593e-7,
        1.46389803181403e-7, -1.07117999898077e-9, 9.69339851919457e-10, -0.000366838537857114,
        -0.000349661240106779, 0.641727204300392, 0.641727204300392, 0.289757289122199, 0.303993370877513,
        0.000629894516203279, 0.000693385617857537, 4.50539854798156e-7, 5.20113587212143e-7,
        2.54589711577036e-10, 3.06571017638641e-10, 0.000792600369225795, 0.000755466976641355,
        0.706708625885607, 0.706708625885607, 0.0231374168420407, 0.0242742059118819, -0.000164870696218446,
        -0.000181489662898668, 4.47106622506182e-7, 4.86305777941202e-7, -7.65822469087791e-10,
        -1.17183559115872e-9, -0.00031896372214657, -0.000304030540092503, 0.707098421865453,
        0.707098421865453, 0.00334108812148205, 0.00350521731985187, -1.49434958428172e-7, -1.6528580633152e-7,
        1.67286989597784e-9, 4.47576208950775e-9, 7.47346854698176e-10, 1.62362437259362e-10,
        -2.66193698941585e-8, -1.91509275076761e-8, 0.000574683318376502, 0.000547834036100508,
        0.707106477692782, 0.707106477692782, 0.000329478151388621, 0.000345645641901939, 1.96303746861486e-8,
        4.28024465755111e-8, 2.14983072339901e-13, 2.32443374786942e-12, 8.59651199007869e-9,
        7.78518361225179e-9, 0.00065987488696318, 0.000629476844630596, 0.707106486859725, 0.707106486859725,
        0.0000195897970471299, 0.0000205443835531918, -3.45422865346826e-11, -9.26923582849814e-11,
        7.65675982886506e-11, 3.45840862726998e-11, 6.51477185924224e-8, 3.71414478337418e-8,
        0.0000796389077169444, 0.0000759703607117922, 0.707106776141595, 0.707106776141595,
        0.000032036235529604, 0.0000336021068620533, 5.97494538498731e-10, 6.11211249916957e-10,
        -5.45429917238791e-13, -1.23497451254794e-13, 1.2050012117632e-6, 1.15208445934766e-6,
        0.707106780903173, 0.707106780903173, 0.0000194893673061455, 0.0000204667890070383,
        3.21331187940999e-11, -1.40726080246176e-10, 7.7806553591738e-12, -2.1689156583936e-12,
        -3.59233453550205e-9, 3.01733195610113e-9, -1.02268242979801e-8, -1.13995526443016e-8,
        3.96828026684986e-6, 3.7627147493377e-6, 0.707106781168255, 0.707106781168255, 3.21169594670978e-6,
        3.39370668688913e-6, -1.22919482172614e-8, 3.80173920453572e-8, 5.72141685444816e-14,
        2.79278022006144e-13, 1.43376484618376e-8, 1.3769838478237e-8, 0.0000184012802906212,
        0.0000175332840888421, 0.7071067809581, 0.7071067809581, 2.04752875575883e-7, 2.83300318191432e-7,
        3.51864527170793e-11, -7.16395072364597e-11, -5.78193786632348e-9, -1.16814080692304e-8,
        6.82881110569883e-8, 8.44843332543935e-8, 0.707106781184068, 0.707106781184068, 1.84623601286592e-6,
        1.8950973318694e-6, -5.09410785654329e-11, 4.52117025327332e-11, 7.15400928836288e-11,
        8.36165321429511e-12, -5.45384570164301e-10, 3.66251816257825e-10, -3.20094801081031e-8,
        -2.62082343582594e-8, 0.707106781186509, 0.707106781186509, 2.30624627204102e-7, 2.2970657771849e-7,
        -3.01416152538345e-13, -2.30247890482769e-12, 2.7199406084563e-9, -9.15289006735626e-10,
        -1.71008893820657e-10, -2.95070400172291e-10, -3.58160491229829e-8, -3.24802430173713e-8,
        0.707106781186544, 0.707106781186544, 6.84964518461775e-8, 6.1271476368654e-8, -3.49864815341556e-11,
        7.67194566022885e-11, -6.99761601668853e-11, -8.55355329992309e-11, 9.34617374737965e-11,
        2.88284186542658e-10, -3.68027626264768e-9, -2.08484800692688e-9, 4.53315691474968e-9,
        3.14179674288276e-8, 0.707106781186546, 0.707106781186546, -4.18123066407728e-8, 2.95262713060843e-8,
        1.29438300344729e-9, -7.65115007732849e-9, 7.88941887770117e-14, 1.0496301090676e-13,
        -1.39561838672051e-8, -1.40528118961327e-8, 0.707106781186546, 0.707106781186546, -3.58720646076126e-8,
        5.6870263647915e-8, -1.98545691133042e-13, 2.668071788877e-11, -3.00438532342435e-10,
        -4.90852491123547e-10, 1.7946946523027e-10, 2.7838180690233e-10, -9.67784115316034e-9,
        -1.44060091534439e-8, 0.707106781186543, 0.707106781186543, -9.71125094503028e-8, -4.46326645159552e-8,
        -3.9611321807321e-12, -4.19928820012122e-12, -2.50489004421586e-10, -2.41020339066207e-9,
        1.66187077944823e-10, -6.70908860124949e-10, -5.08868683220274e-9, 1.97169685341628e-9,
        0.707106781186545, 0.707106781186545, -6.96893548248252e-8, -3.58309204196088e-8, 9.13267802259151e-12,
        5.58682434096704e-11, 2.85389480297907e-9, -1.08026360007823e-8, 1.04874267752594e-10,
        -2.9929351227417e-10, -1.38170009101168e-8, -2.37079156732478e-8, 0.707106781186545, 0.707106781186545,
        -4.2997810072431e-8, -5.02570834567985e-8, -5.20618981804074e-9, -3.201675367854e-8,
        8.47636091819287e-14, 8.73803833884073e-13, 3.39467319176857e-9, 3.35578876426562e-9,
        -2.41967800530949e-10, -2.45956549746563e-8, 0.707106781186547, 0.707106781186547,
        -1.57824972963203e-8, -1.93612546021143e-8, -8.01404055144864e-11, -3.908847882099e-10,
        -7.2019721622073e-11, -2.13145687556826e-10, 2.43095325351385e-10, -8.55457487308606e-10,
        -1.54313576965985e-8, -6.71812913283091e-9, 0.707106781186547, 0.707106781186547, -3.9917080677155e-8,
        -2.46218971016953e-8, -1.9948289718598e-9, -1.16405106756155e-9, -2.50000663808281e-10,
        -3.01973237160376e-10, 2.93642980256251e-10, 1.46251261317827e-10, 6.91966486735684e-9,
        -8.80222229017194e-9, 0.707106781186528, 0.707106781186528, -1.88909063888116e-7, -1.1481255319717e-7,
        -6.95135327012913e-8, 2.61251491376263e-8, -3.16311420260188e-13, -5.29676108209802e-13,
        -2.25081022023164e-9, -1.85743898940963e-9, -1.50613225324547e-8, -2.20690427768062e-8,
        0.707106781186537, 0.707106781186537, -1.38621230763268e-7, -9.51099075649847e-8, 3.83986032468536e-9,
        1.63221049758072e-9, -2.74325106689237e-13, -3.28371403452246e-13, 5.6969326972333e-11,
        2.86456532943005e-9, 1.56390315762179e-9, -1.17083857707883e-8, 0.707106781186546, 0.707106781186546,
        2.47402337538727e-8, 6.71505064215612e-8, -8.048380083531e-11, -3.28452649528428e-11,
        -3.58156870475089e-11, -3.52197147809698e-11, 2.25252286188844e-10, -1.99283423656359e-9,
        1.12525049708505e-9, 1.34896957513362e-8, 0.707106781186547, 0.707106781186547, 3.24969197675388e-9,
        -2.52698912629533e-8, 5.19972975989935e-9, -3.763611724248e-10, 4.37270706455897e-12,
        1.56055650891314e-11, 1.82871738065769e-10, -2.54374511709713e-9, -4.72697533584419e-8,
        -2.86680787160992e-8, 2.38683935632456e-9, -5.25795450562773e-8, 0.707106781186545, 0.707106781186545,
        -1.99524599670017e-8, 1.37719141153723e-8, 1.53577240216403e-9, -1.04226735600505e-9,
        2.19435627461229e-14, 7.51017084129372e-14, 3.26805077325261e-9, -2.59884004076908e-9,
        0.707106781186533, 0.707106781186533, 1.91136571474666e-7, 6.16891324018975e-8, 3.0612523652417e-9,
        1.0336475067192e-9, 9.79107227656152e-11, 4.19219881645427e-11, 1.8239289080115e-10,
        -7.92177240401111e-11});
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
    FG_eval fg_eval;
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

