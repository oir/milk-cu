#ifndef MILK_SMAX_XENT_H
#define MILK_SMAX_XENT_H

namespace milk {
namespace layer {

/* smax_xent layer combines softmax and cross-entropy layers (for simpler backprop)
 * it assumes the output of softmax (h) is backpropped only by the xent function and
 * nothing else. if that assumption doesn't hold for you, use softmax and xent layers
 * separately
 */
template <typename xpu>
class smax_xent : public layer<xpu> {
  public:
    virtual void forward();
    virtual void backward();
    virtual void classify();

    virtual void forward_step(uint t);
    virtual void backward_step(uint t);
    virtual void classify_step(uint t);

    virtual Real loss();     // xent loss value
    virtual Real error();    // misclassification error

    // io
    Data<xpu> h, c;     // h: post-softmax (soft predictions)
                        // c: classifications (hard predictions)

    Input<xpu> x, y; // x: pre-softmax, y: true labels (as scalar not 1-hot vector)

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x, &y}; };
    virtual std::vector<Data<xpu>*> outs() { return {}; };
};

template <typename xpu>
void smax_xent<xpu>::classify() { // make class predictions, assume forward is done
  c.init(x().size(0), 1);
  vec(c()) = argmax(h(), 1);
}

template <typename xpu>
void smax_xent<xpu>::forward() {
  h.init(x().size(0), x().size(1));
  Softmax(h(), x());
  classify();
}

template <typename xpu>
void smax_xent<xpu>::backward() {
  x.d() += h();
  x.d() -= one_hot_encode(vec(y()), h().size(1)); // select above range when padding
}

template <typename xpu>
void smax_xent<xpu>::classify_step(uint t) { // make class predictions, assume forward is done
  if (t == 0) { c.init(x().size(0), 1); c.clone_info(*x); }
  vec(c(t)) = argmax(h(t), 1);
}

template <typename xpu>
void smax_xent<xpu>::forward_step(uint t) {
  if (t == 0) { h.init(x().size(0), x().size(1)); h.clone_info(*x); }
  Softmax(h(t), x(t));
  classify_step(t);
}

template <typename xpu>
void smax_xent<xpu>::backward_step(uint t) {
  x.d(t) += h(t);
  x.d(t) -= one_hot_encode(vec(y(t)), h().size(1)); // select above range when padding
}

template <typename xpu>
Real smax_xent<xpu>::loss() { // loss value (xent). assume forward is done
  VectorContainer<xpu> m(Shape1(h().size(0)));
  m = mat_choose_row_element(F<Log>(h()), vec(y()));
  return -sum(m);
}

template <typename xpu>
Real smax_xent<xpu>::error() { // assume forward and classify done
  VectorContainer<xpu> equals(Shape1(h().size(0)));
  eq(equals, vec(c()), vec(y()));

  return h().size(0) - sum(equals);
}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::smax_xent<xpu>> smax_xent() {
  return std::make_shared<layer::smax_xent<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
