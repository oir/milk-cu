#ifndef MILK_BASE_H
#define MILK_BASE_H

#include <memory>
#include "init.h"
#include "update.h"
#include "utils/shape.h"
#include "utils/dag.h"

namespace milk {

using namespace mshadow;
using namespace mshadow::expr;

template <typename xpu>
class Data {
  public:
    static Stream<xpu>* s;

    std::shared_ptr<MatrixContainer<xpu>> w;    // value
    std::shared_ptr<MatrixContainer<xpu>> grad; // gradient
    std::shared_ptr<Data<xpu>> out = nullptr; // outside of time-range
    uint batch_size = 1;
    std::shared_ptr<sdag> dag = nullptr; // structure info for recursive nets

    Data<xpu>();
    Data<xpu>(uint rows, uint cols);

    uint len() {
      assert((w->size(0) % batch_size) == 0);
      return w->size(0) / batch_size;
    }

    virtual Matrix<xpu>& operator()() { return *w; }
    virtual Matrix<xpu>& d()          { return *grad; }
    // time slices
    virtual Matrix<xpu> operator()(uint t);
    virtual Matrix<xpu> d(uint t);

    virtual void init(uint rows, uint cols);
    virtual void reset_grad();

    virtual bool has_grad() { return grad != nullptr; } // used for truncated bprop
    virtual void clone_info(const Data& other) {
      batch_size = other.batch_size;
      dag = other.dag;
    }
};

template <typename xpu>
Stream<xpu>* Data<xpu>::s = NewStream<xpu>();

template <typename xpu>
std::shared_ptr<MatrixContainer<xpu>> make_MC(uint rows, uint cols,
                                              Real init_val = 0.) {
  auto x = std::make_shared<MatrixContainer<xpu>>(Shape2(rows,cols), init_val);
  x->set_stream(Data<xpu>::s);
  return x;
}

template <typename xpu>
Data<xpu>::Data() { w = make_MC<xpu>(0, 0); }

template <typename xpu>
Data<xpu>::Data(uint rows, uint cols) { w = make_MC<xpu>(rows, cols); }

template <typename xpu>
Matrix<xpu> Data<xpu>::operator()(uint t) {
  if (t >= len()) {
    if (!out or out->w->shape_ != Shape2(batch_size, w->size(1))) {
      out = std::make_shared<Data<xpu>>(batch_size, w->size(1));
      out->reset_grad();
    }
    return (*out)();
  }
  return middle_rows(*w, t*batch_size, batch_size);
}

template <typename xpu>
Matrix<xpu> Data<xpu>::d(uint t) {
  if (t >= len()) return out->d();
  return middle_rows(*grad, t*batch_size, batch_size);
}

template <typename xpu>
void Data<xpu>::init(uint rows, uint cols) {
  if (w->size(0) == rows and w->size(1) == cols) { *w = 0; } // no need to alloc/realloc
  else {
    w->Resize(Shape2(rows,cols), 0.);
    if (grad) grad->Resize(Shape2(rows,cols), 0.);
  }
}

template <typename xpu>
void Data<xpu>::reset_grad() {
  if (!grad or grad->shape_ != w->shape_) {
    grad = make_MC<xpu>(w->size(0), w->size(1));
  }
  else { *grad = 0.; }
  if (out) out->reset_grad();
}

// Input is essentially a Data ptr with additional bookkeeping and
// operators for convenience
template <typename xpu>
class Input {
  public:
    Data<xpu>* in = nullptr;
    uint delay = 0; // for delayed connections when timeslicing in RNNs

    virtual Matrix<xpu>& operator()() { return (*in)(); }
    virtual Matrix<xpu>& d() { return in->d(); }
    virtual Matrix<xpu> operator()(uint t) { return (*in)(t-delay); }
    virtual Matrix<xpu> d(uint t) { return in->d(t-delay); }

    virtual operator bool() { return in; }
    virtual Data<xpu>& operator * () { return *in; }

    virtual void connect_from(Data<xpu>& x, uint a_delay = 0) {
      in = &x; delay = a_delay;
    }
    virtual bool has_grad() { return in->grad != nullptr; } // used for truncated bprop
};

template <typename xpu>
class Weight : public Data<xpu> {
  public:
    std::shared_ptr<updater<xpu>> u;
    void (*initer)(Matrix<xpu>) = init::uniform;

    Real la = defaults::la; // L2 regularizer penalty (shorthand for lambda)

    virtual void init(uint rows, uint cols) {
      this->w = make_MC<xpu>(rows, cols);
      assert(initer);
      initer(*(this->w));
      this->reset_grad();
      if (!u) u = std::make_shared<rmsprop<xpu>>();
      u->init(rows, cols);
      for (auto h : u->history()) h->stream_ = Data<xpu>::s; // TODO: this is prob. not the right place?
    }

    virtual void update() { u->update(*(this->w), *(this->grad)); }
};

} // end namespace milk

#endif
