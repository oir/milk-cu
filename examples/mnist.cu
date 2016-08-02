#include <iostream>
#include <numeric>
#include "../milk.h"

using namespace milk;
using namespace milk::factory;

void mnist_reader(std::string fname,
                  std::vector<Data<gpu>>* X,
                  std::vector<Data<gpu>>* Y,
                  uint rows,
                  uint batch_size = 1) {
  MatrixContainer<cpu> X_(Shape2(rows, 784));
  MatrixContainer<cpu> Y_(Shape2(rows, 1));
  std::ifstream in(fname.c_str());
  assert(in.is_open());
  for (uint i=0; i<rows; i++) {
    in >> Y_[i][0];
    for (uint j=0; j<784; j++)
      in >> X_[i][j];
  }
  X_ *= (1./256.);
  paired_shuffle<cpu>({X_, Y_});
  *X = to_data(X_, batch_size);
  *Y = to_data(Y_, batch_size);
}

int main() {
  InitTensorEngine<gpu>(0);
  Data<gpu>::s = NewStream<gpu>();

  uint batch_size = 64;
  std::vector<Data<gpu>> X, Y, Xtest, Ytest;
  mnist_reader("../../data/mnist/mnist_train.txt", &X, &Y, 60000, batch_size);
  mnist_reader("../../data/mnist/mnist_test.txt", &Xtest, &Ytest, 10000, batch_size);

  std::cout << "Data loaded." << std::endl;

  auto ds = datastream(2);
  auto nn = ff(100,nonlin::tanh()) >>
            ff(100,nonlin::tanh()) >>
            ff(100,nonlin::tanh()) >>
            ff(10,nonlin::id());
  auto loss = smax_xent();
  auto all = ds >> nn >> loss;

  nn->set_updater<adam<gpu>>();

  nn->set_lr(1e-3);
  nn->set_la(1e-4);

  trainer<gpu> t(ds, all);
  for (uint ep=0; ep<10; ep++) {
    t.train({&X, &Y});
    std::cout << t.mean_error({&X, &Y}) << "\t";
    std::cout << t.mean_error({&Xtest, &Ytest}) << std::endl;
  }

  ShutdownTensorEngine<gpu>();
  return 0;
}
