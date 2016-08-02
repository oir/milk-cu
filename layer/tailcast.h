#ifndef MILK_TAILCAST_H
#define MILK_TAILCAST_H

// broadcast tail of a sequence to all of sequence

namespace milk {
namespace layer {

template <typename xpu>
class tailcast : public layer<xpu> {
  public:
    tailcast() {}
    virtual void forward();
    virtual void backward();

    // io
    Data<xpu> h;
    Input<xpu> x;

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
void tailcast<xpu>::forward() {
  h.clone_info(*x.in);
  h.init(x().size(0), x().size(1));
  h.reset_grad();
  uint T = x().size(0) / h.batch_size;
  for (uint t=0; t<T; t++)
    h(t) += bottom_rows(x(), h.batch_size);
}

template <typename xpu>
void tailcast<xpu>::backward() {
  if (x.has_grad()) {
    uint T = x().size(0) / h.batch_size;
    for (uint t=0; t<T; t++)
      bottom_rows(x.d(), h.batch_size) += h.d(t);
  }
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::tailcast<xpu>> tailcast() {
  return std::make_shared<layer::tailcast<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
