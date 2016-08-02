#ifndef MILK_RECURRENT_H
#define MILK_RECURRENT_H

namespace milk {

enum SeqDir {reverse};

namespace layer {

template <typename xpu>
class recurrent : public layer<xpu> {
  public:
    recurrent(int);
    recurrent(int,SeqDir);
    recurrent(int,const Nonlin<xpu>&);
    virtual void forward();
    virtual void backward();
    virtual void init();

    // io
    Data<xpu> h;
    Input<xpu> x;

    // nonlinearity
    Nonlin<xpu> f = nonlin::tanh<xpu>();

    //params
    Weight<xpu> W, V, b;

    int dim = -1;
    int incr = 1; // increment, -1 for reverse direction

    virtual std::vector<Weight<xpu>*> params() { return {&W, &V, &b}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
recurrent<xpu>::recurrent(int a_dim) : dim(a_dim) {}

template <typename xpu>
recurrent<xpu>::recurrent(int a_dim, SeqDir dir)
: dim(a_dim) {
  if (dir == reverse) incr = -1;
}

template <typename xpu>
recurrent<xpu>::recurrent(int a_dim, const Nonlin<xpu>& a_f)
: dim(a_dim), f(a_f) {}

template <typename xpu>
void recurrent<xpu>::init() {
  assert(dim != -1); assert(x.in);
  int xdim = x().size(1);
  W.init(xdim,dim);
  V.init(dim,dim);
  b.init(1, dim);
  b() = 0;
}

template <typename xpu>
void recurrent<xpu>::forward() {
  if (W().size(0) == 0) init();
  h.clone_info(*x);

  uint Tbs = x().size(0);
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  h.init(Tbs, dim);
  h.reset_grad();

  h() = dot(x(), W());
  h() += repmat(vec(b()), Tbs);

  int begin, end; if (incr > 0) { begin=0; end=T; } else { begin=T-1; end=-1; }

  for (int t=begin; t!=end; t+=incr) {
    if (t != begin) h(t) += dot(h(t-incr), V());
    f(h(t), h(t));
  }
}

template <typename xpu>
void recurrent<xpu>::backward() {
  uint Tbs = x().size(0);
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  int begin, end; if (incr > 0) { begin=0; end=T; } else { begin=T-1; end=-1; }

  for (int t=end-incr; t != begin-incr; t-=incr) {
    f.backward(h.d(t), h.d(t), h(t));
    if (t != begin) {
      V.d() += dot(h(t-incr).T(), h.d(t));
      h.d(t-incr) += dot(h.d(t), V().T());
    }
  }

  if (x.has_grad()) // skip if truncation
    x.d()    += dot(h.d(), W().T());
  W.d()      += dot(x().T(), h.d());
  vec(b.d()) += sum_rows(h.d());

  layer<xpu>::backward();
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::recurrent<xpu>> recurrent(uint dim) {
  return std::make_shared<layer::recurrent<xpu>>(dim);
}
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::recurrent<xpu>> recurrent(uint dim, SeqDir dir) {
  return std::make_shared<layer::recurrent<xpu>>(dim,dir);
}
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::recurrent<xpu>> recurrent(uint dim, const Nonlin<xpu>& f) {
  return std::make_shared<layer::recurrent<xpu>>(dim,f);
}
} // end namespace factory

} // end namespace milk

#endif
