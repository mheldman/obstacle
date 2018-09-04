// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "relaxation.h"

namespace py = pybind11;

template<class I, class T, class F>
void _gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
        const I row_start,
         const I row_stop,
         const I row_step
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();

    return gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step
                                 );
}

template<class I, class T, class F>
void _projected_gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<F> & Ax,
       py::array_t<F> & x,
       py::array_t<F> & b,
        const I row_start,
         const I row_stop,
         const I row_step
                             )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const F *_Ax = py_Ax.data();
    F *_x = py_x.mutable_data();
    const F *_b = py_b.data();

    return projected_gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step
                                           );
}

PYBIND11_MODULE(relaxation, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for relaxation.h

    Methods
    -------
    gauss_seidel
    projected_gauss_seidel
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("gauss_seidel", &_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"),
R"pbdoc(
Perform one iteration of Gauss-Seidel relaxation on the linear
 system Ax = b, where A is stored in CSR format and x and b
 are column vectors.

 The unknowns are swept through according to the slice defined
 by row_start, row_end, and row_step.  These options are used
 to implement standard forward and backward sweeps, or sweeping
 only a subset of the unknowns.  A forward sweep is implemented
 with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 number of rows in matrix A.  Similarly, a backward sweep is
 implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).

 Parameters
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     Ax[]       - CSR data array
     x[]        - approximate solution
     b[]        - right hand side
     row_start  - beginning of the sweep
     row_stop   - end of the sweep (i.e. one past the last unknown)
     row_step   - stride used during the sweep (may be negative)

 Returns:
     Nothing, x will be modified in place)pbdoc");

    m.def("projected_gauss_seidel", &_projected_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("projected_gauss_seidel", &_projected_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("projected_gauss_seidel", &_projected_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("projected_gauss_seidel", &_projected_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"),
R"pbdoc(
)pbdoc");

}

