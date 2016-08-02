#ifndef MILK_FF_H
#define MILK_FF_H

namespace milk {
namespace layer {

template <typename xpu>
class ff : public layer<xpu> {
  public:
    ff(int);
    ff(int,const Nonlin<xpu>&);
    virtual void forward();
    virtual void backward();
    virtual void forward_step(uint t);
    virtual void backward_step(uint t);
    virtual void init();

    // io
    Data<xpu> h;
    Input<xpu> x;

    // nonlinearity
    Nonlin<xpu> f = nonlin::tanh<xpu>();

    //params
    Weight<xpu> W, b;

    int dim = -1;

    virtual std::vector<Weight<xpu>*> params() { return {&W, &b}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
ff<xpu>::ff(int a_dim) : dim(a_dim) {}

template <typename xpu>
ff<xpu>::ff(int a_dim, const Nonlin<xpu>& a_f) : dim(a_dim), f(a_f) {}

template <typename xpu>
void ff<xpu>::init() {
  assert(dim != -1); assert(x.in);
  int xdim = x().size(1);
  W.init(xdim,dim);
  b.init(1, dim);
  b() = 0;
}

template <typename xpu>
void ff<xpu>::forward() {
  if (W().size(0) == 0) init();
  h.init(x().size(0), W().size(1));

  h() = dot(x(), W());
  h() += repmat(vec(b()), h().size(0));
  f(h(), h());

  h.reset_grad();
  h.clone_info(*x.in);
}

template <typename xpu>
void ff<xpu>::backward() {
  f.backward(h.d(), h.d(), h());
  if (x.has_grad()) // skip if truncation
    x.d()    += dot(h.d(), W().T());
  W.d()      += dot(x().T(), h.d());
  vec(b.d()) += sum_rows(h.d());

  layer<xpu>::backward();
}

template <typename xpu>
void ff<xpu>::forward_step(uint t) {
  if (W().size(0) == 0) init();
  if (t == 0) {
    h.init(x().size(0), W().size(1));
    h.reset_grad(); h.clone_info(*x);
  }

  h(t) = dot(x(t), W());
  h(t) += repmat(vec(b()), h(t).size(0));
  f(h(t), h(t));
}

template <typename xpu>
void ff<xpu>::backward_step(uint t) {
  f.backward(h.d(t), h.d(t), h(t));
  if (x.has_grad()) // skip if truncation
    x.d(t)    += dot(h.d(t), W().T());
  W.d()       += dot(x(t).T(), h.d(t));
  vec(b.d())  += sum_rows(h.d(t));

  if (t == 0) layer<xpu>::backward();
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::ff<xpu>> ff(uint dim) {
  return std::make_shared<layer::ff<xpu>>(dim);
}
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::ff<xpu>> ff(uint dim, const Nonlin<xpu>& f) {
  return std::make_shared<layer::ff<xpu>>(dim,f);
}
} // end namespace factory

} // end namespace milk

#endif
