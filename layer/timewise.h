#ifndef MILK_TIMEWISE_H
#define MILK_TIMEWISE_H

namespace milk {
namespace layer {

template <typename xpu>
class timewise : public layer<xpu> {
  public:
    std::shared_ptr<layer<xpu>> l;

    timewise(std::shared_ptr<layer<xpu>> a_l);
    virtual void forward();
    virtual void backward();
    virtual void init()     { l->init(); }
    virtual void update()   { l->update(); }

    virtual void set_mode(Mode mode) { l->set_mode(mode); }

    virtual Real error() { return l->error(); }
    virtual Real loss() { return l->loss(); }

    virtual std::vector<Weight<xpu>*> params() { return l->params(); }
    virtual std::vector<Input<xpu>*> ins() { return l->ins(); }
    virtual std::vector<Data<xpu>*> outs() { return l->outs(); }
};

template <typename xpu>
timewise<xpu>::timewise(std::shared_ptr<layer<xpu>> a_l) : l(a_l) {}

template <typename xpu>
void timewise<xpu>::forward() {
  auto& x = *(l->ins()[0]);
  uint Tbs = x().size(0); // time*batch
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  for (int t=0; t<T; t++) l->forward_step(t);
}

template <typename xpu>
void timewise<xpu>::backward() {
  auto& x = *(l->ins()[0]);
  uint Tbs = x().size(0); // time*batch
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  for (int t=T-1; t>=0; t--) l->backward_step(t);
}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev, typename ltype>
std::shared_ptr<layer::timewise<xpu>> timewise(
    std::shared_ptr<ltype> l) {
  return std::make_shared<layer::timewise<xpu>>(l);
}
} // end namespace factory

} // end namespace milk

#endif
