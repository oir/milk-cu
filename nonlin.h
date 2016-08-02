#ifndef MILK_NONLIN_CU
#define MILK_NONLIN_CU

namespace milk {

template <typename xpu>
class Nonlin {
  public:
    void (*forward)(Matrix<xpu> dst, const Matrix<xpu>& x) = nullptr;
    void (*backward)(Matrix<xpu> dst, const Matrix<xpu>& d,
                     const Matrix<xpu>& y) = nullptr;
    void (*backward_add)(Matrix<xpu> dst, const Matrix<xpu>& d,
                     const Matrix<xpu>& y) = nullptr;

    Nonlin(void (*f)(Matrix<xpu>, const Matrix<xpu>&),
           void (*b)(Matrix<xpu>, const Matrix<xpu>&, const Matrix<xpu>&),
           void (*b_add)(Matrix<xpu>, const Matrix<xpu>&, const Matrix<xpu>&))
      : forward(f), backward(b), backward_add(b_add) {}

    void operator()(Matrix<xpu> dst, const Matrix<xpu>& x) {
      forward(dst, x);
    }
};

namespace nonlin {

template <typename xpu>
void tanh_f(Matrix<xpu> dst, const Matrix<xpu>& x) { tanh(dst, x); }
template <typename xpu>
void tanh_b(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst = (1 - y * y) * d;
}
template <typename xpu>
void tanh_b_add(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst += (1 - y * y) * d;
}
template <typename xpu=gpu>
Nonlin<xpu> tanh() { return Nonlin<xpu>(tanh_f, tanh_b, tanh_b_add); }

template <typename xpu>
void sigmoid_f(Matrix<xpu> dst, const Matrix<xpu>& x) { sigmoid(dst, x); }
template <typename xpu>
void sigmoid_b(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst = (1 - y) * y * d;
}
template <typename xpu>
void sigmoid_b_add(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst += (1 - y) * y * d;
}
template <typename xpu=gpu>
Nonlin<xpu> sigmoid() { return Nonlin<xpu>(sigmoid_f, sigmoid_b, sigmoid_b_add); }

template <typename xpu>
void id_f(Matrix<xpu> dst, const Matrix<xpu>& x) { Copy(dst, x); }
template <typename xpu>
void id_b(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  Copy(dst, d);
}
template <typename xpu>
void id_b_add(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst += d;
}
template <typename xpu=gpu>
Nonlin<xpu> id() { return Nonlin<xpu>(id_f, id_b, id_b_add); }

template <typename xpu>
void relu_f(Matrix<xpu> dst, const Matrix<xpu>& x) { relu(dst, x); }
template <typename xpu>
void relu_b(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst = F<IsPositive>(y) * d;
}
template <typename xpu>
void relu_b_add(Matrix<xpu> dst, const Matrix<xpu>& d, const Matrix<xpu>& y) {
  dst += F<IsPositive>(y) * d;
}
template <typename xpu=gpu>
Nonlin<xpu> relu() { return Nonlin<xpu>(relu_f, relu_b, relu_b_add); }

} // end namespace nonlin

} // end namespace milk

#endif
