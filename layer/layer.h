#ifndef MILK_LAYER_H
#define MILK_LAYER_H

#include <fstream>

namespace milk {

enum Mode {TRAIN, TEST};

namespace layer {

template <typename xpu>
class layer {
  public:
    virtual void forward() = 0;
    virtual void backward();

    virtual void forward_step(uint t);  // implement these two to use a layer
    virtual void backward_step(uint t); // inside timewise()

    virtual void init() {};
    virtual void update();
    virtual void reset_grad();

    virtual Real error() { return 0.; }; // loss layers will override this
    virtual Real loss()  { return 0.; }; // loss layers will override this
    Real loss_weight = 1.; // TODO: this is unused for now

    virtual void save_params(std::ostream& out); // TODO: also save/load the
    virtual void load_params(std::istream& in);  //       architecture
    virtual void save_params(const std::string& outfname);
    virtual void load_params(const std::string& infname);

    // TODO: unify all these setters?
    // per Weight. these depend on params() implementation.
    void set_lr(Real lr);
    void set_la(Real la);
    template <typename utype> void set_updater();
    void set_updater(std::function<std::shared_ptr<updater<xpu>>(void)> f); // wonky setter
    void set_initer(void (*initer)(Matrix<xpu>));

    // per layer
    virtual void set_mode(Mode mode) { this->mode = mode; };

    // you have to fill these in for automatic update and resets
    virtual std::vector<Weight<xpu>*> params() = 0;

    // these provide conventions for stacking layers
    virtual std::vector<Input<xpu>*> ins() = 0;
    virtual std::vector<Data<xpu>*> outs() = 0;

    virtual std::vector<Input<xpu>*> dangling_ins();

    virtual uint count_params();

    Mode mode = TRAIN;
};

// base backward only regularizes
template <typename xpu>
void layer<xpu>::backward() {
  for (const auto& W : params())
    if (W->la > 0.)
      W->d() += W->la * (*W)();
}

template <typename xpu>
void layer<xpu>::forward_step(uint t) {
  std::cerr << "Not implemented!" << std::endl;
  assert(false);
}

template <typename xpu>
void layer<xpu>::backward_step(uint t) {
  std::cerr << "Not implemented!" << std::endl;
  assert(false);
}

template <typename xpu>
void layer<xpu>::set_lr(Real lr) {
  for (const auto& W : params())
    W->u->set_lr(lr);
}

template <typename xpu>
void layer<xpu>::set_la(Real la) {
  for (const auto& W : params())
    W->la = la;
}

template <typename xpu> template <typename utype>
void layer<xpu>::set_updater() {
  for (const auto& W : params()) {
    W->u = std::make_shared<utype>();
    if ((*W)().size(0) > 0) W->u->init((*W)().size(0), (*W)().size(1)); //TODO: this is clutter.
  }
}

template <typename xpu>
void layer<xpu>::set_updater(std::function<std::shared_ptr<updater<xpu>>(void)> f) {
  for (const auto& W : params()) {
    W->u = f();
    if ((*W)().size(0) > 0) W->u->init((*W)().size(0), (*W)().size(1)); //TODO: this is clutter.
  }
}

template <typename xpu>
void layer<xpu>::set_initer(void (*initer)(Matrix<xpu>)) {
  for (const auto& W : params())
    W->initer = initer;
}

template <typename xpu>
void layer<xpu>::update() {
  for (const auto& W : params())
    if (W->u->lr > 0.) W->update();
  reset_grad(); //TODO: should i omit this for clarity (explicit reset after updates)?
}

template <typename xpu>
uint layer<xpu>::count_params() {
  uint c = 0;
  for (const auto& W : params())
    c += (*W)().size(0) * (*W)().size(1);
  return c;
}

template <typename xpu>
void layer<xpu>::reset_grad() {
  for (const auto& W : params())
    W->reset_grad();
}

template <typename xpu>
void layer<xpu>::save_params(std::ostream& out) {
  for (auto& W : params()) {
    out << (*W)().size(0) << " " << (*W)().size(1) << std::endl;
    out << (*W)() << std::endl;
    for (auto& h : W->u->history()) {
      out << *h << std::endl;
    }
  }
}

template <typename xpu>
void layer<xpu>::load_params(std::istream& in) {
  for (auto& W : params()) {
    uint rows=0, cols=0;
    in >> rows >> cols;
    W->init(rows, cols);
    in >> (*W)();
    for (auto& h : W->u->history()) { in >> *h; }
  }
}

template <typename xpu>
void layer<xpu>::save_params(const std::string& outfname) {
  std::ofstream out(outfname);
  assert(out.is_open());
  save_params(out);
}

template <typename xpu>
void layer<xpu>::load_params(const std::string& infname) {
  std::ifstream in(infname);
  assert(in.is_open());
  load_params(in);
}

template <typename xpu>
std::vector<Input<xpu>*> layer<xpu>::dangling_ins() {
  auto ins_ = this->ins();
  decltype(ins_) v;
  for (auto& x : ins_) if (!*x) v.push_back(x);
  return v;
}

} // end namespace layer

} // end namespace milk

#endif
