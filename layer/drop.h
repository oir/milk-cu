#ifndef MILK_DROP_H
#define MILK_DROP_H

namespace milk {
namespace layer {

template <typename xpu>
class drop : public layer<xpu> {
  public:
    drop(Real a_p);
    virtual void forward();
    virtual void backward();
    virtual void forward_step(uint t);
    virtual void backward_step(uint t);

    // io
    Data<xpu> h, mask;
    Input<xpu> x;

    static uint seed;
    Real p;

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <>
uint drop<gpu>::seed = 13579;
template <>
uint drop<cpu>::seed = 13579;

template <typename xpu>
drop<xpu>::drop(Real a_p) : p(a_p) {}

template <typename xpu>
void drop<xpu>::forward() {
  assert(x);
  h.init(x().size(0), x().size(1));
  mask.init(x().size(0), x().size(1));
  h.reset_grad();
  h.clone_info(*x);

  if (this->mode == TRAIN) {
    mshadow::Random<xpu, Real>(seed++).SampleUniform(&(mask()), 0., 1.);
    mask() = F<IsNonnegative>(mask() - p);
    h() = x() * mask() * (1./(1.-p));
  } else {
    Copy(h(), x());
  }
}

template <typename xpu>
void drop<xpu>::backward() {
  assert(this->mode == TRAIN);
  if (x.has_grad()) // skip if truncation
    x.d() += h.d() * mask() * (1./(1.-p));
}

template <typename xpu>
void drop<xpu>::forward_step(uint t) {
  if (t == 0) {
    h.init(x().size(0), x().size(1));
    mask.init(x().size(0), x().size(1));
    h.reset_grad();
    h.clone_info(*x);
  }

  if (this->mode == TRAIN) {
    auto mask_t = mask(t);
    mshadow::Random<xpu, Real>(seed++).SampleUniform(&mask_t, 0., 1.);
    mask(t) = F<IsNonnegative>(mask(t) - p);
    h(t) = x(t) * mask(t) * (1./(1.-p));
  } else {
    Copy(h(t), x(t));
  }
}

template <typename xpu>
void drop<xpu>::backward_step(uint t) {
  assert(this->mode == TRAIN);
  if (x.has_grad()) // skip if truncation
    x.d(t) += h.d(t) * mask(t) * (1./(1.-p));
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::drop<xpu>> drop(Real p) {
  return std::make_shared<layer::drop<xpu>>(p);
}
} // end namespace factory

} // end namespace milk

#endif
