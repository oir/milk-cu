#ifndef MILK_UTILS_SHAPE_H
#define MILK_UTILS_SHAPE_H

namespace milk {

template <typename xpu>
Matrix<xpu> middle_rows(Matrix<xpu> x, uint begin, uint rows) { // why not use slice? is this more efficient?
  return Matrix<xpu>(x[begin].dptr_,
                     Shape2(rows, x.size(1)),
                     x.stride_,
                     x.stream_);
}

template <typename xpu>
Matrix<xpu> bottom_rows(Matrix<xpu> x, uint rows) { // why not use slice? is this more efficient?
  return Matrix<xpu>(x[x.size(0)-rows].dptr_,
                     Shape2(rows, x.size(1)),
                     x.stride_,
                     x.stream_);
}

template <typename xpu>
Vector<xpu> vec(Matrix<xpu> x) {
  if (x.size(0) > 1) { // col vector case
    assert(x.size(1) == x.stride_); // assert contiguous if not row vec
    assert(x.size(1) == 1);
    return Vector<xpu>(x.dptr_,
                       Shape1(x.size(0)),
                       x.size(0),
                       x.stream_);
  }
  return Vector<xpu>(x.dptr_,  // row vector case
                     Shape1(x.size(1)),
                     x.stride_,
                     x.stream_);
}

template <typename xpu>
Tensor<xpu, 3, Real> tensor3(Matrix<xpu> x, uint dim) {
  assert(x.size(0) % dim == 0);
  return Tensor<xpu, 3, Real>(x.dptr_,
                              Shape3(dim, x.size(0)/dim, x.size(1)),
                              x.stride_,
                              x.stream_);
}

// simple wrapper to treat a single entry of a tensor as tensor (instead of a
// Real/DType) to use += operator.
// M[i][j] += c; breaks. as_tensor(M[i][j]) += c; works.
// Maybe tweak mshadow so M[i][j] is not a DType but a rank-0 tensor object?
template <typename xpu, typename DType>
Tensor<xpu, 1, DType> as_tensor(DType& data) {
  return Tensor<xpu, 1, DType>(&data, Shape1(1));
}

} // end namespace milk

#endif
