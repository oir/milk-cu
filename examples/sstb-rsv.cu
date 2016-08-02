#include <iostream>
#include <numeric>
#include <queue>
#include <iomanip>
#include "../milk.h"

using namespace milk;
using namespace milk::factory;

std::vector<std::string> split_outermost(const std::string &s) {
  std::vector<std::string> v;
  std::string running;
  int ctr = 0;

  for (uint i=0; i<s.size(); i++) {
    char c = s[i];
    if (c == '(') { ctr++; running.push_back(c); }
    else if (c == ')') { ctr--; running.push_back(c); }
    else if (isspace(c) and ctr == 0) {
      v.push_back(running);
      running = "";
    }
    else { running.push_back(c); }
  }
  v.push_back(running);
  return v;
}

void read_tree(std::string s,
               sdag* dag,
               std::vector<uint>* words,
               std::vector<uint>* labels,
               std::unordered_map<std::string, uint>* w2i,
               std::vector<std::string>* i2w,
               uint internal_pad_value) {
  std::queue<std::pair<uint, std::string>> q;
  q.push(std::make_pair((uint)0, s));

  uint id = 1;

  while (q.size() > 0) {
    auto p = q.front(); q.pop();
    auto i = p.first;
    auto s_ = p.second;
    assert(s_.front() == '(' and s_.back() == ')');
    auto v = split_outermost(s_.substr(1, s_.size()-2));

    if (v.size() == 3) { // internal with two children
      labels->push_back(std::stoi(v[0]));
      words->push_back(internal_pad_value);
      assert(dag->adj_list.size() == i);
      dag->adj_list.push_back({std::make_pair(id, 0),
                               std::make_pair(id+1, 1)});
      auto& left = v[1];
      auto& right = v[2];
      q.push(std::make_pair(id, left));
      q.push(std::make_pair(id+1, right));
      id += 2;
    } else if (v.size() == 2) { // leaf
      labels->push_back(std::stoi(v[0]));
      auto word = v[1];
      if (w2i->find(word) == w2i->end()) {
        (*w2i)[word] = i2w->size();
        i2w->push_back(word);
      }
      words->push_back((*w2i)[word]);
      assert(dag->adj_list.size() == i);
      dag->adj_list.push_back({});
    } else {
      std::cout << v << std::endl;
      assert(false);
    }
  }
}

void print_as_tree(sdag& dag,
                   std::vector<uint>& words,
                   std::vector<std::string>& i2w) {
  std::function<void(uint,std::string)> print_node = [&] (
      uint n, std::string indent) {
    std::cout << indent << " " << i2w[words[n]] << std::endl;
    for (uint i=0; i<dag.children(n).size(); i++)
      print_node(dag.children(n)[i].first, indent+"--");
  };
  print_node(0, "");
}

void sstb_reader(std::string fname,
                 std::vector<Data<gpu>>* X, std::vector<Data<gpu>>* Y,
                 std::unordered_map<std::string, uint>* w2i,
                 std::vector<std::string>* i2w) {
  std::string line;
  std::ifstream in(fname);
  assert(in.is_open());
  while(std::getline(in, line)) {
    std::vector<uint> words, labels;
    auto dag = std::make_shared<sdag>();
    read_tree(line, dag.get(), &words, &labels, w2i, i2w, 0);
    //print_as_tree(dag, words, *i2w);
    uint T = words.size();
    Data<gpu> x(T, 1), y(T, 1);
    x() = 0; y() = 0;
    for (uint t=0; t<T; t++) {
      x()[t] += words[t];
      y()[t] += labels[t];
    }
    x.dag = y.dag = dag;
    x.batch_size = y.batch_size = 1;
    X->push_back(x);
    Y->push_back(y);
  }
}

Real root_error(std::vector<std::vector<Data<gpu>>*> dataset,
                std::shared_ptr<layer::datastream<gpu>> ds,
                std::shared_ptr<layer::layer<gpu>> nn,
                std::shared_ptr<layer::smax_xent<gpu>> top) {
  nn->set_mode(TEST);
  Real err=0, tot=0;
  do {
    nn->forward();
    VectorContainer<gpu> equals(Shape1(top->h.batch_size));
    eq(equals, vec(top->c(0)), vec(top->y(0)));
    err += top->h.batch_size - sum(equals);
    tot += top->h.batch_size;
  } while (ds->count != 0);
  return err/tot;
}

int main() {
  InitTensorEngine<gpu>();
  std::cout << std::setprecision(3);

  std::vector<Data<gpu>> X, Y, Xdev, Ydev, Xtest, Ytest;
  std::unordered_map<std::string, uint> w2i;
  std::vector<std::string> i2w;

  std::cout << "reading..." << std::endl;

  i2w.push_back("**");
  w2i[""] = 0;

  sstb_reader("../../data/sstb/trees/train.txt", &X,     &Y,     &w2i, &i2w);
  sstb_reader("../../data/sstb/trees/dev.txt",   &Xdev,  &Ydev,  &w2i, &i2w);
  sstb_reader("../../data/sstb/trees/test.txt",  &Xtest, &Ytest, &w2i, &i2w);

  uint N = w2i.size();

  auto ds = datastream(2);
  auto wv = proj(300, N+1);
  auto top = smax_xent();
  auto nn = ds >> wv >> recursive(50, 2, nonlin::relu()) >>
                        recursive(50, 2, nonlin::relu()) >>
                        recursive(50, 2, nonlin::relu()) >>
                        ff(5,nonlin::id()) >> top;

  wv->init();
  std::cout << "loading wvecs... " << std::flush;
  load_wv_table("/home/oirsoy/glove.840B.300d.txt", 300, &(wv->W()), w2i);
  std::cout << "done." << std::endl;

  nn->set_updater<adagrad<gpu>>();
  nn->set_lr(1e-2);
  wv->set_lr(0.); // keep wv table fixed
  nn->set_la(1e-5);

  trainer<gpu> t(ds, nn);

  for (uint ep=0; ep<30; ep++) {
    t.train({&X, &Y});
    std::cout << 1.-t.mean_error({&X, &Y}) << "\t";
    std::cout << 1.-t.mean_error({&Xdev, &Ydev}) << "\t";
    std::cout << 1.-root_error({&Xdev, &Ydev}, ds, nn, top) << std::endl;
  }

  ShutdownTensorEngine<gpu>();
  return 0;
}
