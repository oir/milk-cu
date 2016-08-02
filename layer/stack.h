#ifndef MILK_STACK_H
#define MILK_STACK_H

namespace milk {
namespace layer {

template <typename xpu>
class stack : public layer<xpu> {
  public:
    std::shared_ptr<layer<xpu>> top, bottom;

    stack(std::shared_ptr<layer<xpu>> a_bottom,
          std::shared_ptr<layer<xpu>> a_top);

    virtual void forward()  { bottom->forward(); top->forward(); };
    virtual void backward() { top->backward();   bottom->backward(); };
    virtual void forward_step(uint t)  { bottom->forward_step(t); top->forward_step(t); };
    virtual void backward_step(uint t) { top->backward_step(t);   bottom->backward_step(t); };
    virtual void init()     { bottom->init();    top->init(); };
    virtual void update()   { bottom->update();  top->update(); };

    virtual void set_mode(Mode mode) {
      bottom->set_mode(mode); top->set_mode(mode);
    }
    virtual Real error() { return bottom->error() + top->error(); }
    virtual Real loss()  { return bottom->loss()  + top->loss();  }

    virtual std::vector<Weight<xpu>*> params() {
      return bottom->params() + top->params();
    }
    virtual std::vector<Input<xpu>*> ins();
    virtual std::vector<Data<xpu>*> outs();
};

template <typename xpu>
stack<xpu>::stack(std::shared_ptr<layer<xpu>> a_bottom,
                  std::shared_ptr<layer<xpu>> a_top)
  : bottom(a_bottom), top(a_top) {
  auto ins = top->ins();
  auto outs = bottom->outs();

  // EXPERIMENTAL: skip if in-connection is already connected. not thoroughly tested.
  for (uint i=0,j=0; i<ins.size() and j<outs.size(); ) {
    if (ins[i]->in) { i++; continue; }
    ins[i]->connect_from(*outs[j]);
    i++; j++;
  }
}

template <typename xpu>
std::vector<Input<xpu>*> stack<xpu>::ins() {
  auto bottom_ins = bottom->ins();
  uint outsize = bottom->outs().size();
  auto top_ins = top->ins();
  for (uint i=outsize; i<top_ins.size(); i++)
    bottom_ins.push_back(top_ins[i]);
  return bottom_ins;
}

template <typename xpu>
std::vector<Data<xpu>*> stack<xpu>::outs() {
  auto top_outs = top->outs();
  uint insize = top->ins().size();
  auto bottom_outs = bottom->outs();
  for (uint i=insize; i<bottom_outs.size(); i++)
    top_outs.push_back(bottom_outs[i]);
  return top_outs;
}

} // end namespace layer

namespace factory {
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::stack<xpu>> stack(
    std::shared_ptr<layer::layer<xpu>> bottom,
    std::shared_ptr<layer::layer<xpu>> top) {
  return std::make_shared<layer::stack<xpu>>(bottom, top);
}
} // end namespace factory

template <typename xpu, template <typename> class ltype,
          template <typename> class ltype2>
std::shared_ptr<layer::stack<xpu>> operator>>(
    std::shared_ptr<ltype<xpu>> bottom,
    std::shared_ptr<ltype2<xpu>> top) {
  return std::make_shared<layer::stack<xpu>>(bottom, top);
}

} // end namespace milk

#endif
