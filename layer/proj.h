#ifndef MILK_PROJ_H
#define MILK_PROJ_H

#include <unordered_set>

namespace milk {
namespace layer {

template <typename xpu>
class proj : public layer<xpu> {
  public:
    proj(int,int);
    virtual void forward();
    virtual void backward();
    virtual void forward_step(uint t);
    virtual void backward_step(uint t);
    virtual void init();

    // io
    Data<xpu> h;
    Input<xpu> x;

    //params
    Weight<xpu> W;

    int dim = -1;
    int size = -1; // vocab size

    virtual std::vector<Weight<xpu>*> params() { return {&W}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
proj<xpu>::proj(int a_dim, int a_size)
: dim(a_dim), size(a_size) {}

template <typename xpu>
void proj<xpu>::init() {
  assert(dim != -1);
  W.init(size,dim);
}

template <typename xpu>
void proj<xpu>::forward() {
  if (W().size(0) == 0) init();
  assert(x.in);

  uint Tbs = x().size(0);
  h.init(Tbs,dim);

  h() += take(vec(x()), W());

  h.reset_grad();
  h.clone_info(*x.in);
  Matrix<xpu> h = *(W.u->history()[0]);
}

template <typename xpu>
void proj<xpu>::backward() {
  if (W.u->lr > 0) AddTakeGrad(W.d(), vec(x()), h.d());
  W.d() += (W.la * W() * F<IsNonzero>(W.d())); // L2-regularize iff it is used
  //layer<xpu>::backward();
}

template <typename xpu>
void proj<xpu>::forward_step(uint t) {
  if (W().size(0) == 0) init();
  if (t == 0) {
    assert(x.in);
    uint Tbs = x().size(0);
    h.init(Tbs,dim);
    h.reset_grad();
    h.clone_info(*x);
  }

  h(t) += take(vec(x(t)), W());
}

template <typename xpu>
void proj<xpu>::backward_step(uint t) {
  if (W.u->lr > 0) AddTakeGrad(W.d(), vec(x(t)), h.d(t));
  if (t == 0) W.d() += (W.la * W() * F<IsNonzero>(W.d())); // L2-regularize iff it is used
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::proj<xpu>> proj(uint dim, uint size) {
  return std::make_shared<layer::proj<xpu>>(dim,size);
}
} // end namespace factory

} // end namespace milk

#endif
