#ifndef MILK_RECURSIVE_H
#define MILK_RECURSIVE_H

namespace milk {

namespace layer {

template <typename xpu>
class recursive : public layer<xpu> {
  public:
    recursive(int, int);
    recursive(int, int, const Nonlin<xpu>&);
    virtual void forward();
    virtual void backward();
    virtual void init();

    // io
    Data<xpu> h;
    Input<xpu> x;

    // nonlinearity
    Nonlin<xpu> f = nonlin::tanh<xpu>();

    //params
    Weight<xpu> W, b;
    std::vector<Weight<xpu>> V;  // recursive weights per edge label

    int dim = -1;

    virtual std::vector<Weight<xpu>*> params() {
      return get_ptrs(V) + std::vector<Weight<xpu>*>({&W, &b});
    };
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
recursive<xpu>::recursive(int a_dim, int N) : dim(a_dim) { V.resize(N); }

template <typename xpu>
recursive<xpu>::recursive(int a_dim, int N, const Nonlin<xpu>& a_f)
: dim(a_dim), f(a_f) { V.resize(N); }

template <typename xpu>
void recursive<xpu>::init() {
  assert(dim != -1); assert(x);
  int xdim = x().size(1);
  W.init(xdim,dim);
  for (auto& v : V) v.init(dim,dim);
  b.init(1, dim);
  b() = 0;
}

template <typename xpu>
void recursive<xpu>::forward() {
  if (W().size(0) == 0) init();
  auto& dag = h.dag;

  h.clone_info(*x);
  h.init(x().size(0), dim);
  h.reset_grad();

  h() = dot(x(), W());
  h() += repmat(vec(b()), x().size(0));

  for (int n=dag->size()-1; n>=0; n--) {
    for (auto& p : dag->children(n)) {
      uint i = p.first, l = p.second;
      h(n) += dot(h(i), V[l]());
    }
    f(h(n), h(n));
  }
}

template <typename xpu>
void recursive<xpu>::backward() {
  auto& dag = h.dag;

  for (uint n=0; n<dag->size(); n++) {
    f.backward(h.d(n), h.d(n), h(n));
    for (auto& p : dag->children(n)) {
      uint i = p.first, l = p.second;
      V[l].d() += dot(h(i).T(), h.d(n));
      h.d(i) += dot(h.d(n), V[l]().T());
    }
  }

  if (x.has_grad())
    x.d()    += dot(h.d(), W().T());
  W.d()      += dot(x().T(), h.d());
  vec(b.d()) += sum_rows(h.d());

  layer<xpu>::backward();
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::recursive<xpu>> recursive(uint dim, uint N) {
  return std::make_shared<layer::recursive<xpu>>(dim,N);
}
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::recursive<xpu>> recursive(uint dim, uint N,
                                                 const Nonlin<xpu>& f) {
  return std::make_shared<layer::recursive<xpu>>(dim,N,f);
}
} // end namespace factory

} // end namespace milk

#endif
