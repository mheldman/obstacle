// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "monotone_restriction.h"

namespace py = pybind11;

template<class I, class F>
void _monotone_restrict_2d(
 py::array_t<F> & ucoarse,
   py::array_t<F> & ufine,
               const I mx,
               const I my,
         const I mxcoarse,
         const I mycoarse
                           )
{
    auto py_ucoarse = ucoarse.mutable_unchecked();
    auto py_ufine = ufine.unchecked();
    F *_ucoarse = py_ucoarse.mutable_data();
    const F *_ufine = py_ufine.data();

    return monotone_restrict_2d<I, F>(
                 _ucoarse, ucoarse.shape(0),
                   _ufine, ufine.shape(0),
                       mx,
                       my,
                 mxcoarse,
                 mycoarse
                                      );
}

PYBIND11_MODULE(monotone_restriction, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for monotone_restriction.h

    Methods
    -------
    monotone_restrict_2d
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("monotone_restrict_2d", &_monotone_restrict_2d<int, float>,
        py::arg("ucoarse").noconvert(), py::arg("ufine").noconvert(), py::arg("mx"), py::arg("my"), py::arg("mxcoarse"), py::arg("mycoarse"));
    m.def("monotone_restrict_2d", &_monotone_restrict_2d<int, double>,
        py::arg("ucoarse").noconvert(), py::arg("ufine").noconvert(), py::arg("mx"), py::arg("my"), py::arg("mxcoarse"), py::arg("mycoarse"),
R"pbdoc(
Parameters  FIXME
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

}

