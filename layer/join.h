#ifndef MILK_JOIN_H
#define MILK_JOIN_H

namespace milk {
namespace layer {

template <typename xpu>
class join : public layer<xpu> {
  public:
    std::shared_ptr<layer<xpu>> left, right;

    join(std::shared_ptr<layer<xpu>> a_left,
         std::shared_ptr<layer<xpu>> a_right);

    virtual void forward()  { left->forward();   right->forward(); };
    virtual void backward() { right->backward(); left->backward(); };
    virtual void forward_step(uint t)  { left->forward_step(t);   right->forward_step(t); };
    virtual void backward_step(uint t) { right->backward_step(t); left->backward_step(t); };
    virtual void init()     { left->init();      right->init(); };
    virtual void update()   { left->update();    right->update(); };

    virtual void set_mode(Mode mode) {
      left->set_mode(mode); right->set_mode(mode);
    }
    virtual Real error() { return left->error() + right->error(); }
    virtual Real loss()  { return left->loss()  + right->loss();  }

    virtual std::vector<Weight<xpu>*> params() {
      return left->params() + right->params();
    }
    virtual std::vector<Input<xpu>*> ins() {
      return left->ins() + right->ins();
    }
    virtual std::vector<Data<xpu>*> outs() {
      return left->outs() + right->outs();
    }
};

template <typename xpu>
join<xpu>::join(std::shared_ptr<layer<xpu>> a_left,
                std::shared_ptr<layer<xpu>> a_right)
  : left(a_left), right(a_right) {}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::join<xpu>> join(
    std::shared_ptr<layer::layer<xpu>> left,
    std::shared_ptr<layer::layer<xpu>> right) {
  return std::make_shared<layer::join<xpu>>(left, right);
}
} // end namespace factory

template <typename xpu, template <typename> class ltype,
          template <typename> class ltype2>
std::shared_ptr<layer::join<xpu>> operator,(
    std::shared_ptr<ltype<xpu>> left,
    std::shared_ptr<ltype2<xpu>> right) {
  return std::make_shared<layer::join<xpu>>(left, right);
}

} // end namespace milk

#endif
