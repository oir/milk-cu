#ifndef MILK_TAIL_H
#define MILK_TAIL_H

namespace milk {
namespace layer {

template <typename xpu>
class tail : public layer<xpu> {
  public:
    tail() {}
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
void tail<xpu>::forward() {
  h.clone_info(*x.in);
  h.init(h.batch_size, x().size(1));
  h() += bottom_rows(x(), h.batch_size);
  h.reset_grad();
}

template <typename xpu>
void tail<xpu>::backward() {
  if (x.has_grad()) bottom_rows(x.d(), h.batch_size) += h.d();
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::tail<xpu>> tail() {
  return std::make_shared<layer::tail<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
