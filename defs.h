#ifndef Real
#define Real float
#endif

#ifndef uint
#define uint unsigned
#endif

#ifndef MilkDefaultDev
#define MilkDefaultDev gpu
#endif

namespace milk {

template <typename xpu>
using Matrix = mshadow::Tensor<xpu, 2, Real>;
template <typename xpu>
using MatrixContainer = mshadow::TensorContainer<xpu, 2, Real>;
template <typename xpu>
using Vector = mshadow::Tensor<xpu, 1, Real>;
template <typename xpu>
using VectorContainer = mshadow::TensorContainer<xpu, 1, Real>;


namespace defaults {

Real lr = 1e-3; // learning rate
Real la = 0.;   // lambda (L2 regularizer penalty)
Real clip = 5.; // clip value for update rules

} // end namespace defaults
} // end namespace milk
