#ifndef MILK_CAT_H
#define MILK_CAT_H

// concatenation layer

namespace milk {
namespace layer {

template <typename xpu>
class cat : public layer<xpu> {
  public:
    cat() {};
    virtual void forward();
    virtual void backward();

    // io
    Data<xpu> h;
    Input<xpu> x1, x2;

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x1, &x2}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
void cat<xpu>::forward() {
  uint dim1 = x1().size(1);
  uint dim2 = x2().size(1);
  uint dim = dim1 + dim2;
  h.init(x1().size(0), dim);

  slice<1>(h(),0,dim1)   = x1();
  slice<1>(h(),dim1,dim) = x2();

  h.reset_grad();
  h.clone_info(*x1);
}

template <typename xpu>
void cat<xpu>::backward() {
  uint dim1 = x1().size(1);
  uint dim2 = x2().size(1);
  uint dim = dim1 + dim2;
  if (x1.has_grad()) // skip if truncation
    x1.d() += slice<1>(h.d(),0,dim1);
  if (x2.has_grad()) // skip if truncation
    x2.d() += slice<1>(h.d(),dim1,dim);
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::cat<xpu>> cat() {
  return std::make_shared<layer::cat<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
