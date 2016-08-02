#include <iostream>
#include <numeric>
#include "../milk.h"

using namespace milk;
using namespace milk::factory;

void sstb_reader(std::string fname,
                 std::vector<Data<cpu>>* X, std::vector<Data<cpu>>* Y,
                 std::unordered_map<std::string, uint>* w2i,
                 std::vector<std::string>* i2w) {
  std::string line;
  std::ifstream in(fname);
  assert(in.is_open());
  while(std::getline(in, line)) {
    auto v = split(line, '\t');
    uint label = std::stoi(v[1]);
    auto sent = split(v[0], ' ');

    Data<cpu> x, y;
    x.init(sent.size(), 1);
    for (uint t=0; t<sent.size(); t++) {
      auto& w = sent[t];
      if (w2i->find(w) == w2i->end()) {
        (*w2i)[w] = i2w->size();
        i2w->push_back(w);
      }
      x()[t][0] = (*w2i)[w];
    }

    y.init(1,1);
    y()[0][0] = label;

    X->push_back(x);
    Y->push_back(y);
  }
}

int main() {
  InitTensorEngine<gpu>();

  uint batch_size = 32;
  std::vector<Data<cpu>> X, Y, Xdev, Ydev, Xtest, Ytest;
  std::vector<Data<gpu>> Xb, Yb, Xbdev, Ybdev, Xbtest, Ybtest;
  std::unordered_map<std::string, uint> w2i;
  std::vector<std::string> i2w;

  std::cout << "reading..." << std::endl;

  sstb_reader("../../data/sstb_flat/train_root.txt", &X,     &Y,     &w2i, &i2w);
  sstb_reader("../../data/sstb_flat/dev_root.txt",   &Xdev,  &Ydev,  &w2i, &i2w);
  sstb_reader("../../data/sstb_flat/test_root.txt",  &Xtest, &Ytest, &w2i, &i2w);

  uint N = w2i.size();

  std::cout << "batching..." << std::endl;

  batch_seq_single_label(&Xb, &Yb, X, Y, N, batch_size);
  batch_seq_single_label(&Xbdev, &Ybdev, Xdev, Ydev, N, batch_size);
  batch_seq_single_label(&Xbtest, &Ybtest, Xtest, Ytest, N, batch_size);

  auto ds = datastream(2);
  auto wv = proj(300, N+1);
  auto nn = ds >> wv >> lstm(50) >> tail() >> ff(5,nonlin::id()) >> smax_xent();

  wv->init();
  std::cout << "loading wvecs... " << std::flush;
  load_wv_table("/home/oirsoy/glove.840B.300d.txt", 300, &(wv->W()), w2i);
  std::cout << "done." << std::endl;
  wv->W()[N] = 0;

  nn->set_updater<rmsprop<gpu>>();
  nn->set_lr(1e-3);
  wv->set_lr(0.); // keep wv table fixed
  nn->set_la(1e-3);

  trainer<gpu> t(ds, nn);

  for (uint e=0; e<100; e++) {
    Real running_err = t.train({&Xb, &Yb});
    Real dev_err = t.mean_error({&Xbdev, &Ybdev});
    std::cout << 1.-running_err << " " << 1.-dev_err << std::endl;
  }

  ShutdownTensorEngine<gpu>();
  return 0;
}
