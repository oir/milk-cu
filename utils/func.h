#ifndef MILK_UTILS_FUNC_H
#define MILK_UTILS_FUNC_H

namespace milk {

using namespace mshadow;
using namespace mshadow::expr;

struct Floor {
  MSHADOW_XINLINE static Real Map(Real x) { return std::floor(x); }
};

struct Tanh {
  MSHADOW_XINLINE static Real Map(Real x) {
    if (x < 0) {
      Real e = exp(2.*x);
      return (e-1.) / (e+1.);
    } else {
      Real e = exp(-2.*x);
      return (1.-e) / (1.+e);
    }
  }
};
template<typename xpu>
void tanh(Matrix<xpu> out, const Matrix<xpu> &in) { out = F<Tanh>(in); }

struct Sigmoid {
  MSHADOW_XINLINE static Real Map(Real x) { return 1. / (1. + exp(-x)); }
};
template<typename xpu>
void sigmoid(Matrix<xpu> out, const Matrix<xpu> &in) {
  out = F<Sigmoid>(in);
}

struct Relu {
  MSHADOW_XINLINE static Real Map(Real x) { return (x > 0.) ? x : 0.; }
};
template<typename xpu>
void relu(Matrix<xpu> out, const Matrix<xpu> &in) {
  out = F<Relu>(in);
}

struct Log { MSHADOW_XINLINE static Real Map(Real x) { return std::log(x); } };
template<typename xpu>
void log(Matrix<xpu> out, const Matrix<xpu> &in) { out = F<Log>(in); }

struct Sqrt { MSHADOW_XINLINE static Real Map(Real x) { return std::sqrt(x); } };

#define CLIP 5. // TODO: make this non-hardcoded
struct Clip {
  MSHADOW_XINLINE static Real Map(Real x) {
    Real e = (x > CLIP) ? CLIP : x;
    return ((e < -CLIP) ? -CLIP : e);
  }
};
template<typename xpu>
void clip(Matrix<xpu> out, const Matrix<xpu> &in) {
  out = F<Clip>(in);
}

struct Eq {
  MSHADOW_XINLINE static Real Map(Real a, Real b) { return (a == b); }
};
template <typename xpu, int dim, typename DType>
void eq(Tensor<xpu, dim, DType> out, const Tensor<xpu, dim, DType> &a,
        const Tensor<xpu, dim, DType> &b) {
  out = F<Eq>(a, b);
}

struct Geq {
  MSHADOW_XINLINE static Real Map(Real a, Real b) { return (a >= b); }
};
template <typename xpu, int dim, typename DType>
void geq(Tensor<xpu, dim, DType> out, const Tensor<xpu, dim, DType> &a,
         const Tensor<xpu, dim, DType> &b) {
  out = F<Geq>(a, b);
}

struct IsNonzero {
  MSHADOW_XINLINE static Real Map(Real a) { return (a != 0.); }
};

struct IsPositive {
  MSHADOW_XINLINE static Real Map(Real a) { return (a > 0.); }
};


template <typename xpu>
Real sum(Vector<xpu> m) {
  // is there really no sum routine in mshadow?
  Matrix<xpu> as_column(m.dptr_, Shape2(m.size(0),1));
  VectorContainer<xpu> s(Shape1(1));
  as_column.stride_ = 1;
  s = sum_rows(as_column);
  VectorContainer<cpu> s_(Shape1(1));
  Copy(s_, s);
  return s_[0];
}

template <typename xpu>
Real sum(Matrix<xpu> m) {
  // is there really no sum routine in mshadow?
  VectorContainer<xpu> rs(Shape1(m.size(1)));
  rs = sum_rows(m);
  return sum(rs);
}

template <typename xpu>
Real sqsum(Matrix<xpu> m) {
  MatrixContainer<xpu> m_(m.shape_);
  m_ = m * m;
  return sum(m_);
}

template<int dimkeep,  typename SrcExp, typename DType, int etype>
inline ReduceTo1DExp<SrcExp, DType, red::maximum,
       ExpInfo<SrcExp>::kDim - dimkeep>
maxall_except_dim(const Exp<SrcExp, DType, etype> &exp) {
  return ReduceTo1DExp<SrcExp, DType, red::maximum,
                       ExpInfo<SrcExp>::kDim - dimkeep>(exp.self(), DType(1));
}

template<typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<red::maximum, SrcExp, DType, ExpInfo<SrcExp>::kDim, true,
 ExpInfo<SrcExp>::kDim - 1>
argmax(const Exp<SrcExp, DType, etype> &src, int axis) {
 return reduce_with_axis<red::maximum, true>(src.self(), axis);
}



} // end namespace milk

#endif
