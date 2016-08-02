#define Real double
#define MilkDefaultDev cpu
#include "milk.h"

using namespace milk;
using namespace milk::factory;

class Stats {         // small class to keep track of abs/rel diffs and other
  public:             // stuff that might be of interest when gradchecking
    Real max_abs_diff = 0;
    Real max_rel_diff = 0; // TODO

    void accumulate(Real analytic, Real numeric) {
      Real abs = std::abs(analytic - numeric);
      max_abs_diff = std::max(max_abs_diff, abs);
    }

    void accumulate(const Stats& s) {
      max_abs_diff = std::max(max_abs_diff, s.max_abs_diff);
    }

    void print() {
      std::cout << "Max Abs Diff: " << max_abs_diff << std::endl;
    }
};

template<typename xpu, template <typename> class ltype>
Stats check_grad_wrt(std::shared_ptr<ltype<xpu>> l,
                     Data<xpu>*                  W,
                     std::function<void(void)>   init_delta,
                     std::function<Real(void)>   obj_fn,
                     uint                        verbosity=0) {
  Real eps = 1e-4;
  Stats s;

  W->reset_grad();
  l->forward();
  init_delta();
  l->backward();

  Matrix<xpu>& M = (*W)();
  Matrix<xpu>& dM = W->d();

  MatrixContainer<cpu> M_(M.shape_), dM_(dM.shape_);
  Copy(M_, M); Copy(dM_, dM);

  for (uint i=0; i<M.size(0); i++) {
    for (uint j=0; j<M.size(1); j++) {
      as_tensor<xpu>(M[i][j]) += eps;
      l->forward();
      Real up = obj_fn();
      Copy(as_tensor<xpu>(M[i][j]), as_tensor<cpu>(M_[i][j]));
      as_tensor<xpu>(M[i][j]) -= eps;
      l->forward();
      Real down = obj_fn();
      Real numeric_grad = (up - down) / (2*eps);

      if (verbosity > 0)
        std::cout << dM_[i][j] << "\t" << numeric_grad
                  << std::endl;

      s.accumulate(dM_[i][j], numeric_grad);
      Copy(as_tensor<xpu>(M[i][j]), as_tensor<cpu>(M_[i][j]));
    }
  }
  if (verbosity > 0) std::cout << std::endl;
  return s;
}

template <typename xpu, template <typename> class ltype>
void check_grad(std::shared_ptr<ltype<xpu>> l, uint verbosity=0) {
  uint xdim = 4;
  uint T = 5;
  uint bs = 2;

  auto ins = l->dangling_ins();
  std::vector<Data<xpu>> xs(ins.size());

  for (uint i=0; i<ins.size(); i++) {
    auto& x = xs[i];
    x.init(bs*T,xdim);
    mshadow::Random<xpu, Real>(i).SampleUniform(&(x()), -10., 10.);
    x.reset_grad();
    x.batch_size = bs;
    ins[i]->connect_from(x);
  }

  l->forward(); // this is to init weights

  for (auto W : l->params()) (*W)() *= 10;

  auto obj_fn = [&]() { // this will be specialized (mainly for loss layers)
    Real val = 0;
    for (auto& y : l->outs()) val += sqsum((*y)());
    return val + l->loss();
  };
  auto init_delta = [&]() { // this will also be specialized
    for (auto& y : l->outs()) y->d() += 2 * (*y)();
  };

  Stats s;
  for (auto W : l->params())
    s.accumulate(check_grad_wrt(l, W, init_delta, obj_fn, verbosity));
  for (auto& x : xs)
    s.accumulate(check_grad_wrt(l, &x, init_delta, obj_fn, verbosity));
  s.print();
}

template <typename xpu>
void check_grad(std::shared_ptr<layer::smax_xent<xpu>> l, uint verbosity=0) {
  Data<xpu> x, y;
  x.init(2,3); y.init(2,1); y() = 1;
  l->x.connect_from(x); l->y.connect_from(y);
  mshadow::Random<xpu, Real>(0).SampleUniform(&(x()), -10., 10.);

  auto init_delta = []() {}; // no external error
  auto obj_fn = [&]() { return l->loss(); }; // xent loss value is the objective fn

  auto s = check_grad_wrt(l, &x, init_delta, obj_fn, verbosity);
  s.print();
};

template <typename xpu>
void check_grad(std::shared_ptr<layer::cf_smax_xent<xpu>> l, uint verbosity=0) {
  Data<xpu> x1, x2, y;
  x1.init(2,3); x2.init(2,2); y.init(2,1);
  x1.reset_grad(); x2.reset_grad();
  y() = 0; y()[0] += 1; y()[1] += 5;
  l->x1.connect_from(x1);
  l->x2.connect_from(x2);
  l->y.connect_from(y);

  mshadow::Random<xpu, Real>(0).SampleUniform(&(x1()), -10., 10.);
  mshadow::Random<xpu, Real>(1).SampleUniform(&(x2()), -10., 10.);

  auto init_delta = []() {}; // no external error
  auto obj_fn = [&]() { return l->loss(); }; // xent loss value is the objective fn

  auto s = check_grad_wrt(l, &x1, init_delta, obj_fn, verbosity);
  s.accumulate(check_grad_wrt(l, &x2, init_delta, obj_fn, verbosity));
  s.print();
};

template <typename xpu>
void check_grad(std::shared_ptr<layer::sqerr<xpu>> l, uint verbosity=0) {
  Data<xpu> x, y;
  x.init(2,3); y.init(2,3);
  l->x.connect_from(x); l->y.connect_from(y);
  mshadow::Random<xpu, Real>(0).SampleUniform(&(x()), -10., 10.);
  mshadow::Random<xpu, Real>(1).SampleUniform(&(y()), -10., 10.);

  auto init_delta = []() {}; // no external error
  auto obj_fn = [&]() { return l->loss(); }; // xent loss value is the objective fn

  auto s = check_grad_wrt(l, &x, init_delta, obj_fn, verbosity);
  s.print();
};

template <typename xpu>
void check_grad(std::shared_ptr<layer::proj<xpu>> l, uint verbosity=0) {
  Data<xpu> x;
  x.init(4, 1);
  as_tensor<xpu>(x()[0][0]) = 0;
  as_tensor<xpu>(x()[1][0]) = 2;
  as_tensor<xpu>(x()[2][0]) = 0;
  as_tensor<xpu>(x()[3][0]) = 5;
  l->x.connect_from(x);
  l->forward(); // to init weights
  l->W() *= 10;

  auto init_delta = [&]() { l->h.d() += 2 * l->h(); };
  auto obj_fn     = [&]() { return sqsum(l->h()); };
  check_grad_wrt(l, &(l->W), init_delta, obj_fn, verbosity).print();
}

template <typename xpu>
void check_grad(std::shared_ptr<layer::recursive<xpu>> l, uint verbosity=0) {
  Data<xpu> x(10, 2);
  x.batch_size = 2;
  mshadow::Random<xpu, Real>(0).SampleUniform(&(x()), -10., 10.);
  auto dag = std::make_shared<sdag>();
  dag->adj_list = { {{1,0}, {2,1}}, {}, {{3,0}, {4,1}}, {}, {} };
  //dag->labels = { {0, 1}, {}, {0, 1}, {}, {} };
  x.dag = dag;
  assert(l->V.size() > 1);

  l->x.connect_from(x);
  l->forward(); // to init weights

  auto init_delta = [&]() { l->h.d() += 2 * l->h(); };
  auto obj_fn     = [&]() { return sqsum(l->h()); };
  check_grad_wrt(l, &(l->W), init_delta, obj_fn, verbosity).print();
}

#define CHECK_GRAD(layer)                                    \
std::cout << "Checking " << #layer << std::endl;             \
check_grad(layer, verbosity);                                \
std::cout << std::endl;                                      \

int main(int argc, char** argv) {
  InitTensorEngine<MilkDefaultDev>();
  uint verbosity = 0;
  if (argc > 1) verbosity = stoi(std::string(argv[1]));

  CHECK_GRAD( ff(3) )
  CHECK_GRAD( ff(3) >> ff(2) )
  CHECK_GRAD( smax_xent() )
  CHECK_GRAD( sqerr() )
  CHECK_GRAD( recurrent(3) )
  CHECK_GRAD( recurrent(3,reverse) )
  CHECK_GRAD( cat() )
  CHECK_GRAD( cast() )
  CHECK_GRAD( tail() )
  CHECK_GRAD( tailcast() )
  CHECK_GRAD( (ff(3), ff(2)) )
  CHECK_GRAD( cast()
              >> (recurrent(3), recurrent(2,reverse))
              >> cat() )
  //CHECK_GRAD( proj(3, 5) )

  CHECK_GRAD( lstm(3) )
  CHECK_GRAD( cf_smax_xent() )

  CHECK_GRAD( timewise(lstm(3)) )
  CHECK_GRAD( timewise(lstm(3) >> lstm(2)) )
  CHECK_GRAD( timewise(ff(3)) )

  CHECK_GRAD( recursive(3,2) )

  ShutdownTensorEngine<MilkDefaultDev>();

  return 0;
}
