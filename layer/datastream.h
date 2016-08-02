#ifndef MILK_DATASTREAM_H
#define MILK_DATASTREAM_H

#include <numeric>

namespace milk {

namespace layer {

template <typename xpu>
class datastream : public layer<xpu> {
  public:
    datastream(uint n=2);
    datastream(const std::vector<std::vector<Data<xpu>>*>&);
    virtual void forward();
    virtual void backward() {}
    virtual void init();
    virtual void set_data(const std::vector<std::vector<Data<xpu>>*>&);

    //output
    std::vector<Data<xpu>> x;

    // data storage
    std::vector<std::vector<Data<xpu>>*> X; // can't just use Matrix because of sequence data

    // for curriculum learning
    uint max_len = std::numeric_limits<uint>::max();

    virtual std::vector<Weight<xpu>*> params() { return {}; }
    virtual std::vector<Input<xpu>*> ins() { return {}; }
    virtual std::vector<Data<xpu>*> outs() { return get_ptrs(x); }

    // state
    uint count = 0;
    std::vector<uint> perm;
};

/* n is the number of data components, e.g:
 *   2 for classification / regression (in & out, X & Y)
 *   1 for unsupervised learning (X)
 *   >=2 for multimodal learning (X1 & X2 & Y, etc)
 */
template <typename xpu>
datastream<xpu>::datastream(uint n) { x.resize(n); }

template <typename xpu>
datastream<xpu>::datastream(
    const std::vector<std::vector<Data<xpu>>*>& datalist) {
  x.resize(datalist.size());
  set_data(datalist);
}

template <typename xpu>
void datastream<xpu>::init() {
  assert(X.size() > 0);                        //
  assert(X[0]->size() != 0);                   // data is loaded
  for (uint i=1; i<X.size(); i++)
    assert(X[i]->size() == X[i-1]->size());    // # of instances match

  //perm.resize(X[0]->size());
  //std::iota(perm.begin(), perm.end(), 0);              // 0..N
  // playing with curriculum learning...
  perm.clear();
  for (uint j=0; j<X[0]->size(); j++) if ((*X[0])[j].len() <= max_len)
      perm.push_back(j);
  count = 0;
}

template <typename xpu>
void datastream<xpu>::set_data(
    const std::vector<std::vector<Data<xpu>>*>& datalist) {
  assert(datalist.size() > 0);
  X = datalist;
  perm.clear(); // force a reset of permutation, init() will be called again
}

template <typename xpu>
void datastream<xpu>::forward() {
  if (perm.size() == 0) init();
  if (count == 0 && this->mode == TRAIN)
    std::random_shuffle(perm.begin(), perm.end());
  uint j = perm[count];
  for (uint i=0; i<X.size(); i++)
    x[i] = (*X[i])[j]; // these should already have their batch_size in them
  count++;
  if (count == perm.size()) count = 0;
}

} // end namespace layer

namespace factory {

template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::datastream<xpu>> datastream(uint n=2) {
  return std::make_shared<layer::datastream<xpu>>(n);
}
template <typename xpu=MilkDefaultDev>
std::shared_ptr<layer::datastream<xpu>> datastream(
    const std::vector<std::vector<Data<xpu>>*>& datalist) {
  return std::make_shared<layer::datastream<xpu>>(datalist);
}

} // end namespace factory

} // end namespace milk

#endif
