#ifndef MILK_LSTM_H
#define MILK_LSTM_H

namespace milk {

//enum SeqDir {reverse};

namespace layer {

template <typename xpu>
class lstm : public layer<xpu> {
  public:
    lstm(int);
    //lstm(int,SeqDir,Real,Real);
    virtual void forward();
    virtual void backward();
    virtual void forward_step(uint t);
    virtual void backward_step(uint t);
    virtual void init();

    // io
    Data<xpu> h;
    Input<xpu> x;

    // tmps
    Data<xpu> i, f, c, o, g, h_;

    // nonlinearity
    Nonlin<xpu> nl_gate = nonlin::sigmoid<xpu>();
    Nonlin<xpu> nl_g = nonlin::tanh<xpu>();
    Nonlin<xpu> nl_h = nonlin::tanh<xpu>();

    //params
    Weight<xpu> Wix, Wfx, Wcx, Wox,
                Wih, Wfh, Wch, Woh,
                Wic, Wfc,      Woc,
                bi,  bf,  bc,  bo;

    int dim;
    int incr = 1; // increment, -1 for reverse direction

    virtual std::vector<Weight<xpu>*> params() { return {&Wix, &Wfx, &Wcx, &Wox,
                                                         &Wih, &Wfh, &Wch, &Woh,
                                                         &Wic, &Wfc,       &Woc,
                                                         &bi,  &bf,  &bc,  &bo }; }
    virtual std::vector<Input<xpu>*> ins() { return {&x}; };
    virtual std::vector<Data<xpu>*> outs() { return {&h}; };
};

template <typename xpu>
lstm<xpu>::lstm(int a_dim) : dim(a_dim) {}

template <typename xpu>
void lstm<xpu>::init() {
  assert(x.in);
  int xdim = x().size(1);
  for (auto& w : {&Wix, &Wfx, &Wcx, &Wox}) w->init(xdim,dim);
  for (auto& w : {&Wih, &Wfh, &Wch, &Woh, &Wic, &Wfc, &Woc}) w->init(dim,dim);
  for (auto& w : {&bi, &bf, &bc, &bo}) { w->init(1,dim); (*w)() = 0; }
}

template <typename xpu>
void lstm<xpu>::forward() {
  if (Wix().size(0) == 0) init();
  uint Tbs = x().size(0); // time*batch
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;
  int begin, end; if (incr > 0) { begin=0; end=T; } else { begin=T-1; end=-1; }

  for (auto& w : {&i, &f, &g, &c, &o, &h_, &h}) {
    w->init(Tbs,dim); w->clone_info(*x.in); w->reset_grad();
  }
  i() = dot(x(), Wix()); i() += repmat(vec(bi()), Tbs);
  f() = dot(x(), Wfx()); f() += repmat(vec(bf()), Tbs);
  g() = dot(x(), Wcx()); g() += repmat(vec(bc()), Tbs);
  o() = dot(x(), Wox()); o() += repmat(vec(bo()), Tbs);

  for (int t=begin; t!=end; t+=incr) {
    if (t != begin) {
      i(t) += dot(h(t-incr), Wih());
      i(t) += dot(c(t-incr), Wic());
      f(t) += dot(h(t-incr), Wfh());
      f(t) += dot(c(t-incr), Wfc());
    }
    nl_gate(i(t), i(t));
    nl_gate(f(t), f(t));
    if (t != begin)
      g(t) += dot(h(t-incr), Wch());
    nl_g(g(t), g(t));

    c(t) = i(t) * g(t);
    if (t != begin) c(t) += (1.-f(t)) * c(t-incr);

    if (t != begin) {
      o(t) += dot(h(t-incr), Woh());
      o(t) += dot(c(t-incr), Woc());
    }
    nl_gate(o(t), o(t));

    nl_h(h_(t), c(t));
    h(t) = o(t) * h_(t);
  }
}

template <typename xpu>
void lstm<xpu>::backward() {
  uint Tbs = x().size(0);
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  int begin, end; if (incr > 0) { begin=0; end=T; } else { begin=T-1; end=-1; }

  for (int t=end-incr; t != begin-incr; t-=incr) {
    h_.d(t) += h.d(t) * o(t);
    o.d(t)  += h.d(t) * h_(t);
    nl_h.backward_add(c.d(t), h_.d(t), h_(t));

    nl_gate.backward(o.d(t), o.d(t), o(t));
    if (t != begin) {
      h.d(t-incr) += dot(o.d(t), Woh().T());
      Woh.d()     += dot(h(t-incr).T(), o.d(t));
      c.d(t-incr) += dot(o.d(t), Woc().T());
      Woc.d()     += dot(c(t-incr).T(), o.d(t));
    }

    if (t != begin) {
      f.d(t) -= c.d(t) * c(t-incr);
      c.d(t-incr) += c.d(t) * (1.-f(t));
    }
    i.d(t) += c.d(t) * g(t);
    g.d(t) += c.d(t) * i(t);

    nl_g.backward(g.d(t), g.d(t), g(t));
    if (t != begin) {
      h.d(t-incr) += dot(g.d(t), Wch().T());
      Wch.d()     += dot(h(t-incr).T(), g.d(t));
    }

    nl_gate.backward(f.d(t), f.d(t), f(t));
    nl_gate.backward(i.d(t), i.d(t), i(t));

    if (t != begin) {
      c.d(t-incr) += dot(f.d(t), Wfc().T());
      Wfc.d()     += dot(c(t-incr).T(), f.d(t));
      h.d(t-incr) += dot(f.d(t), Wfh().T());
      Wfh.d()     += dot(h(t-incr).T(), f.d(t));
      c.d(t-incr) += dot(i.d(t), Wic().T());
      Wic.d()     += dot(c(t-incr).T(), i.d(t));
      h.d(t-incr) += dot(i.d(t), Wih().T());
      Wih.d()     += dot(h(t-incr).T(), i.d(t));
    }
  }

  Wix.d() += dot(x().T(), i.d());
  Wfx.d() += dot(x().T(), f.d());
  Wcx.d() += dot(x().T(), g.d());
  Wox.d() += dot(x().T(), o.d());
  vec(bi.d()) += sum_rows(i.d());
  vec(bf.d()) += sum_rows(f.d());
  vec(bc.d()) += sum_rows(g.d());
  vec(bo.d()) += sum_rows(o.d());

  if (x.has_grad()) {
    x.d() += dot(i.d(), Wix().T());
    x.d() += dot(f.d(), Wfx().T());
    x.d() += dot(g.d(), Wcx().T());
    x.d() += dot(o.d(), Wox().T());
  }

  layer<xpu>::backward();
}

template <typename xpu>
void lstm<xpu>::forward_step(uint t) {
  if (Wix().size(0) == 0) init();
  uint Tbs = x().size(0); // time*batch
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;
  int begin; if (incr > 0) { begin=0; } else { begin=T-1; }

  if (t == begin) for (auto& w : {&i, &f, &g, &c, &o, &h_, &h}) {
    w->init(Tbs,dim); w->clone_info(*x.in); w->reset_grad();
  }

  i(t) = dot(x(t), Wix()); i(t) += repmat(vec(bi()), bs);
  f(t) = dot(x(t), Wfx()); f(t) += repmat(vec(bf()), bs);
  g(t) = dot(x(t), Wcx()); g(t) += repmat(vec(bc()), bs);
  o(t) = dot(x(t), Wox()); o(t) += repmat(vec(bo()), bs);

  if (t != begin) {
    i(t) += dot(h(t-incr), Wih());
    i(t) += dot(c(t-incr), Wic());
    f(t) += dot(h(t-incr), Wfh());
    f(t) += dot(c(t-incr), Wfc());
  }
  nl_gate(i(t), i(t));
  nl_gate(f(t), f(t));
  if (t != begin)
    g(t) += dot(h(t-incr), Wch());
  nl_g(g(t), g(t));

  c(t) = i(t) * g(t);
  if (t != begin) c(t) += (1.-f(t)) * c(t-incr);

  if (t != begin) {
    o(t) += dot(h(t-incr), Woh());
    o(t) += dot(c(t-incr), Woc());
  }
  nl_gate(o(t), o(t));

  nl_h(h_(t), c(t));
  h(t) = o(t) * h_(t);
}

template <typename xpu>
void lstm<xpu>::backward_step(uint t) {
  uint Tbs = x().size(0);
  uint bs = x.in->batch_size;
  uint T = Tbs / bs;

  int begin; if (incr > 0) { begin=0; } else { begin=T-1; }

  h_.d(t) += h.d(t) * o(t);
  o.d(t)  += h.d(t) * h_(t);
  nl_h.backward_add(c.d(t), h_.d(t), h_(t));

  nl_gate.backward(o.d(t), o.d(t), o(t));
  if (t != begin) {
    h.d(t-incr) += dot(o.d(t), Woh().T());
    Woh.d()     += dot(h(t-incr).T(), o.d(t));
    c.d(t-incr) += dot(o.d(t), Woc().T());
    Woc.d()     += dot(c(t-incr).T(), o.d(t));
  }

  if (t != begin) {
    f.d(t) -= c.d(t) * c(t-incr);
    c.d(t-incr) += c.d(t) * (1.-f(t));
  }
  i.d(t) += c.d(t) * g(t);
  g.d(t) += c.d(t) * i(t);

  nl_g.backward(g.d(t), g.d(t), g(t));
  if (t != begin) {
    h.d(t-incr) += dot(g.d(t), Wch().T());
    Wch.d()     += dot(h(t-incr).T(), g.d(t));
  }

  nl_gate.backward(f.d(t), f.d(t), f(t));
  nl_gate.backward(i.d(t), i.d(t), i(t));

  if (t != begin) {
    c.d(t-incr) += dot(f.d(t), Wfc().T());
    Wfc.d()     += dot(c(t-incr).T(), f.d(t));
    h.d(t-incr) += dot(f.d(t), Wfh().T());
    Wfh.d()     += dot(h(t-incr).T(), f.d(t));
    c.d(t-incr) += dot(i.d(t), Wic().T());
    Wic.d()     += dot(c(t-incr).T(), i.d(t));
    h.d(t-incr) += dot(i.d(t), Wih().T());
    Wih.d()     += dot(h(t-incr).T(), i.d(t));
  }

  Wix.d() += dot(x(t).T(), i.d(t));
  Wfx.d() += dot(x(t).T(), f.d(t));
  Wcx.d() += dot(x(t).T(), g.d(t));
  Wox.d() += dot(x(t).T(), o.d(t));
  vec(bi.d()) += sum_rows(i.d(t));
  vec(bf.d()) += sum_rows(f.d(t));
  vec(bc.d()) += sum_rows(g.d(t));
  vec(bo.d()) += sum_rows(o.d(t));

  if (x.has_grad()) {
    x.d(t) += dot(i.d(t), Wix().T());
    x.d(t) += dot(f.d(t), Wfx().T());
    x.d(t) += dot(g.d(t), Wcx().T());
    x.d(t) += dot(o.d(t), Wox().T());
  }

  if (t == begin) layer<xpu>::backward();
}

} // end namespace layer


namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::lstm<xpu>> lstm(uint dim) {
  return std::make_shared<layer::lstm<xpu>>(dim);
}
} // end namespace factory

} // end namespace milk

#endif
