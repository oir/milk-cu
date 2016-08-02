#ifndef MILK_UPDATE_H
#define MILK_UPDATE_H

#include "utils/func.h"
#include "base.h"

namespace milk {

template <typename xpu>
class updater {
  public:
    Real lr = defaults::lr; // not every updater has learning rate but most do
    virtual void set_lr(Real a_lr) { lr = a_lr; }

    virtual void update(Matrix<xpu> w, Matrix<xpu> g) = 0;
    virtual void init(uint rows, uint cols) = 0;
    virtual std::vector<MatrixContainer<xpu>*> history() = 0;
};

template <typename xpu>
class adagrad : public updater<xpu> {
  public:
    Real eps = 1e-6;
    MatrixContainer<xpu> h;

    std::vector<MatrixContainer<xpu>*> history() { return {&h}; }
    void init(uint rows, uint cols) {
      h = MatrixContainer<xpu>(Shape2(rows,cols), 0.);
    }
    void update(Matrix<xpu> w, Matrix<xpu> g) {
      clip(g, g); // is it okay to override g? are we sure g won't be used elsewhere?
      h += g * g;
      w -= this->lr * g / F<Sqrt>(h + eps);
    }
};

template <typename xpu>
class rmsprop : public updater<xpu> {
  public:
    Real eps = 1e-6, rho = 0.9;
    MatrixContainer<xpu> h;

    std::vector<MatrixContainer<xpu>*> history() { return {&h}; }
    void init(uint rows, uint cols) {
      h = MatrixContainer<xpu>(Shape2(rows,cols), 0.);
    }
    void update(Matrix<xpu> w, Matrix<xpu> g) {
      clip(g, g);
      h = h * rho + g * g * (1.-rho);
      w -= this->lr * g / F<Sqrt>(h + eps);
    }
};

template <typename xpu>
class momentum : public updater<xpu> {
  public:
    Real rho = 0.9;
    MatrixContainer<xpu> v;

    std::vector<MatrixContainer<xpu>*> history() { return {&v}; }
    void init(uint rows, uint cols) {
      v = MatrixContainer<xpu>(Shape2(rows,cols), 0.);
    }
    void update(Matrix<xpu> w, Matrix<xpu> g) {
      clip(g, g);
      v = v * rho + this->lr * g;
      w -= v;
    }
};

template <typename xpu>
class adam : public updater<xpu> {
  public:
    Real epsh = 1e-8, beta1 = 0.9, beta2 = 0.999;
    Real beta1_t = 1., beta2_t = 1.;

    MatrixContainer<xpu> m, v;

    std::vector<MatrixContainer<xpu>*> history() { return {&m, &v}; }
    void init(uint rows, uint cols) {
      m = MatrixContainer<xpu>(Shape2(rows,cols), 0.);
      v = MatrixContainer<xpu>(Shape2(rows,cols), 0.);
    }
    void update(Matrix<xpu> w, Matrix<xpu> g) {
      clip(g, g);
      beta1_t *= beta1; beta2_t *= beta2;
      m = beta1 * m + (1.-beta1) * g;
      v = beta2 * v + (1.-beta2) * g * g;
      Real alpha_t = this->lr * std::sqrt(1.-beta2_t) / (1.-beta1_t);
      w -= alpha_t * m / (F<Sqrt>(v) + epsh);
    }
};

//} // end namespace update

} // end namespace milk

#endif
