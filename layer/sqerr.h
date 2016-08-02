#ifndef MILK_SQERR_H
#define MILK_SQERR_H

namespace milk {
namespace layer {

template <typename xpu>
class sqerr : public layer<xpu> {
  public:
    virtual void forward() {}
    virtual void backward();
    virtual Real loss();     // 0.5 * squared error
    virtual Real error();    // squared error

    // io
    Input<xpu> x, y; // x: predicted, y: true

    virtual std::vector<Weight<xpu>*> params() { return {}; };
    virtual std::vector<Input<xpu>*> ins() { return {&x, &y}; };
    virtual std::vector<Data<xpu>*> outs() { return {}; };
};

template <typename xpu>
void sqerr<xpu>::backward() {
  assert(x and y);
  x.d() += (x() - y());
}

template <typename xpu>
Real sqerr<xpu>::loss() { // loss value (0.5 sqerr). assume forward is done
  MatrixContainer<xpu> d(x().shape_, 0.);
  d += x() - y();
  return 0.5 * sqsum(d);
}

template <typename xpu>
Real sqerr<xpu>::error() { // most times we are interested in sqerr, not (0.5 * sqerr)
  return 2 * loss();
}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::sqerr<xpu>> sqerr() {
  return std::make_shared<layer::sqerr<xpu>>();
}
} // end namespace factory

} // end namespace milk

#endif
