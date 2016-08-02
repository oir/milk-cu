#ifndef MILK_CAST_H
#define MILK_CAST_H

// broadcast layer to assign single output to multiple inputs

namespace milk {
namespace layer {

template <typename xpu>
class cast : public layer<xpu> {
  public:
    cast(uint n=2);
    virtual void forward();
    virtual void backward() {}
    virtual void forward_step(uint t) { if (t == 0) this->forward(); }
    virtual void backward_step(uint t) {}

    // io
    std::vector<Data<xpu>> h;
    Input<xpu> x;

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return get_ptrs(h); };
};

template <typename xpu>
cast<xpu>::cast(uint n) { h.resize(n); }

template <typename xpu>
void cast<xpu>::forward() {
  for (auto& h_ : h)
    h_ = *x; // no copy, just ptr+metadata assignment
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::cast<xpu>> cast(uint n=2) {
  return std::make_shared<layer::cast<xpu>>(n);
}
} // end namespace factory

} // end namespace milk

#endif
