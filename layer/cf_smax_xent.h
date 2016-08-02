#ifndef MILK_CF_SMAX_XENT_H
#define MILK_CF_SMAX_XENT_H

// Class-factored softmax + cross-entropy loss

namespace milk {
namespace layer {

template <typename xpu>
class cf_smax_xent : public layer<xpu> {
  public:
    virtual void forward();
    virtual void backward();
    virtual void classify();
    virtual void forward_step(uint t);
    virtual void backward_step(uint t);
    virtual void classify_step(uint t);
    virtual Real loss();    // xent loss value
    virtual Real error();   // misclassification error

    // io
    Data<xpu> h1, h2, c;   // h: post-softmax (soft predictions)
                           // c: class predictions
    Input<xpu> x1, x2, y;  // x: pre-softmax, y: true labels (as scalar not 1-hot vector)

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x1, &x2, &y}; };
    virtual std::vector<Data<xpu>*> outs() { return {}; };
};

template <typename xpu>
void cf_smax_xent<xpu>::classify() {
  c.init(x1().size(0), 1);
  const uint& dim2 = x2().size(1);
  vec(c()) = argmax(h1(),1) * dim2 + argmax(h2(), 1);
}

template <typename xpu>
void cf_smax_xent<xpu>::forward() {
  h1.init(x1().size(0), x1().size(1));
  h2.init(x2().size(0), x2().size(1));
  Softmax(h1(), x1());
  Softmax(h2(), x2());
  classify();
}

template <typename xpu>
void cf_smax_xent<xpu>::backward() {
  uint dim1 = x1().size(1);
  uint dim2 = x2().size(1);

  MatrixContainer<xpu> y1(y().shape_, 0.), y2(y().shape_, 0.);
  y1.stream_ = y2.stream_ = y().stream_;
  y1 = F<Floor>(y() / dim2);
  y2 = y() - y1*dim2;

  if (x1.has_grad()) {
    x1.d() += h1();
    x1.d() -= one_hot_encode(vec(y1), dim1);
  }
  if (x2.has_grad()) {
    x2.d() += h2();
    x2.d() -= one_hot_encode(vec(y2), dim2);
  }
}

template <typename xpu>
Real cf_smax_xent<xpu>::loss() { // loss value (xent). assume forward is done
  uint dim1 = x1().size(1);
  uint dim2 = x2().size(1);

  MatrixContainer<xpu> y1(y().shape_, 0.), y2(y().shape_, 0.);
  y1.stream_ = y2.stream_ = y().stream_;
  y1 = F<Floor>(y() / dim2);
  y2 = y() - y1*dim2;

  VectorContainer<xpu> m1(Shape1(h1().size(0))), m2(Shape1(h2().size(0)));
  m1 = mat_choose_row_element(F<Log>(h1()), vec(y1));
  m2 = mat_choose_row_element(F<Log>(h2()), vec(y2));
  return -sum(m1)-sum(m2);
}

template <typename xpu>
Real cf_smax_xent<xpu>::error() {
  VectorContainer<xpu> equals(Shape1(h1().size(0)));
  eq(equals, vec(c()), vec(y()));
  return h1().size(0) - sum(equals);
}

template <typename xpu>
void cf_smax_xent<xpu>::classify_step(uint t) {
  if (t == 0) { c.init(x1().size(0), 1); c.clone_info(*x1); }
  const uint& dim2 = x2().size(1);
  vec(c(t)) = argmax(h1(t),1) * dim2 + argmax(h2(t), 1);
}

template <typename xpu>
void cf_smax_xent<xpu>::forward_step(uint t) {
  if (t == 0) {
    h1.init(x1().size(0), x1().size(1)); h2.init(x2().size(0), x2().size(1));
    h1.clone_info(*x1); h2.clone_info(*x2);
  }
  Softmax(h1(t), x1(t));
  Softmax(h2(t), x2(t));
  classify_step(t);
}

template <typename xpu>
void cf_smax_xent<xpu>::backward_step(uint t) {
  assert(y);
  uint dim1 = x1().size(1);
  uint dim2 = x2().size(1);

  MatrixContainer<xpu> y1(y(t).shape_, 0.), y2(y(t).shape_, 0.);
  y1.stream_ = y2.stream_ = y().stream_;
  y1 = F<Floor>(y(t) / dim2);
  y2 = y(t) - y1*dim2;


  if (x1.has_grad()) {
    x1.d(t) += h1(t);
    x1.d(t) -= one_hot_encode(vec(y1), dim1);
  }
  if (x2.has_grad()) {
    x2.d(t) += h2(t);
    x2.d(t) -= one_hot_encode(vec(y2), dim2);
  }
}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::cf_smax_xent<xpu>> cf_smax_xent() {
  return std::make_shared<layer::cf_smax_xent<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
