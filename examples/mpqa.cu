#include <iostream>
#include <iomanip>
#include <numeric>
#include <set>
#include "../milk.h"

using namespace milk;
using namespace milk::factory;

// returns soft (precision, recall, F1)
// counts proportional overlap & binary overlap
template <typename xpu>
std::vector<Real> Fscore(std::shared_ptr<layer::layer<xpu>> layers,
                         std::shared_ptr<layer::datastream<xpu>> ds,
                         std::shared_ptr<layer::smax_xent<xpu>> top,
                         std::vector<std::string>& i2y,
                         std::vector<Data<xpu>> &sents,
                         std::vector<Data<xpu>> &labels,
                         Real label_pad_value) {
  timer::tic("F score");
  ds->set_data({&sents, &labels});
  uint nExprPredicted = 0;
  uint nExprTrue = 0;
  Real precNumerProp = 0, precNumerBin = 0;
  Real recallNumerProp = 0, recallNumerBin = 0;
  for (uint i=0; i<sents.size(); i++) { // per sentence batch
    layers->forward();

    auto& ds_x = ds->x[0];
    auto& ds_y = ds->x[1];

    MatrixContainer<cpu> x(ds_x().shape_);
    MatrixContainer<cpu> label(ds_y().shape_);
    MatrixContainer<cpu> amax(top->c().shape_);

    Copy(x,     ds_x());
    Copy(amax,  top->c());
    Copy(label, ds_y());

    uint bs = ds_x.batch_size;

    for (uint b=0; b<bs; b++) {
      std::vector<std::string> labelsPredicted, labelsTrue;

      uint T = x.size(0) / bs;
      for (uint j=0; j<T; j++) {
        if (label[bs*j + b][0] != label_pad_value) { // if not padding
          uint maxi = amax[bs*j + b][0];
          labelsPredicted.push_back(i2y[maxi]); // B, I or O
          uint idx = label[bs*j + b][0];
          labelsTrue.push_back(i2y[idx]);
        }
      }

      std::string y, t, py="", pt="";
      std::vector<std::pair<uint,uint> > pred, tru;
      int l1=-1, l2=-1;

      for (uint j=0; j<labelsTrue.size(); j++) { // per token in a sentence
        t = labelsTrue[j];
        y = labelsPredicted[j];

        if (t == "B") {
          if (l1 != -1)
            tru.push_back(std::make_pair(l1,j));
          l1 = j;
        } else if (t == "I") {
          assert(l1 != -1);
        } else if (t == "O") {
          if (l1 != -1)
            tru.push_back(std::make_pair(l1,j));
          l1 = -1;
        } else
          assert(false);

        if ((y == "B") || ((y == "I") && ((py == "") || (py == "O")))) {
          nExprPredicted++;
          if (l2 != -1)
            pred.push_back(std::make_pair(l2,j));
          l2 = j;
        } else if (y == "I") {
          assert(l2 != -1);
        } else if (y == "O") {
          if (l2 != -1)
            pred.push_back(std::make_pair(l2,j));
          l2 = -1;
        } else {
          std::cout << y << std::endl;
          assert(false);
        }

        py = y;
        pt = t;
      }
      if ((l1 != -1) && (l1 != labelsTrue.size()))
        tru.push_back(std::make_pair(l1,labelsTrue.size()));
      if ((l2 != -1) && (l2 != labelsTrue.size()))
        pred.push_back(std::make_pair(l2,labelsTrue.size()));

      auto trum = std::vector<bool>(tru.size(),false);
      auto predm = std::vector<bool>(pred.size(),false);
      for (uint a=0; a<tru.size(); a++) {
        std::pair<uint,uint> truSpan = tru[a];
        nExprTrue++;
        for (uint b=0; b<pred.size(); b++) {
          std::pair<uint,uint> predSpan = pred[b];

          uint lmax, rmin;
          if (truSpan.first > predSpan.first)
            lmax = truSpan.first;
          else
            lmax = predSpan.first;
          if (truSpan.second < predSpan.second)
            rmin = truSpan.second;
          else
            rmin = predSpan.second;

          uint overlap = 0;
          if (rmin > lmax)
            overlap = rmin-lmax;
          if (predSpan.second == predSpan.first)
            std::cout << predSpan.first << std::endl;
          assert(predSpan.second != predSpan.first);
          precNumerProp += (Real)overlap/(predSpan.second-predSpan.first);
          recallNumerProp += (Real)overlap/(truSpan.second-truSpan.first);
          if (!predm[b] && overlap > 0) {
            precNumerBin += (Real)(overlap>0);
            predm[b] = true;
          }
          if (!trum[a] && overlap>0) {
            recallNumerBin += 1;
            trum[a]=true;
          }
        }
      }
    }
  }
  Real precisionProp = (nExprPredicted==0) ? 1 : precNumerProp/nExprPredicted;
  Real recallProp = (nExprTrue==0) ? 1 : recallNumerProp/nExprTrue;
  Real f1Prop = (precisionProp+recallProp) == 0 ? 0 :
                   (2*precisionProp*recallProp)/(precisionProp+recallProp);
  Real precisionBin = (nExprPredicted==0) ? 1 : precNumerBin/nExprPredicted;
  Real recallBin = (nExprTrue==0) ? 1 : recallNumerBin/nExprTrue;
  Real f1Bin = (2*precisionBin*recallBin)/(precisionBin+recallBin);

  timer::toc("F score");
  return {precisionProp, recallProp, f1Prop,
          precisionBin,  recallBin,  f1Bin};
}

// TODO: fix conventions, T here is for Tag, but I also have L for Label
void readSentences(std::vector<Data<cpu>>&                  X,
                   std::vector<Data<cpu>>&                  T,
                   std::unordered_map<std::string,uint>&  w2i,
                   std::vector<std::string>&              i2w,
                   std::unordered_map<std::string,uint>&  y2i,
                   std::vector<std::string>&              i2y,
                   std::string                          fname) {
  std::ifstream in(fname.c_str());
  assert(in.is_open());
  std::string line;
  std::vector<uint> x, t; // individual sentences and labels
  while(std::getline(in, line)) {
    if (is_whitespace(line) or in.eof()) {
      if (x.size() != 0) {
        Data<cpu> x_(x.size(), 1), t_(t.size(), 1);
        for (uint i=0; i<x.size(); i++) { x_(i) = x[i]; t_(i) = t[i]; }
        X.push_back(x_);
        T.push_back(t_);
        x.clear();
        t.clear();
      }
    } else {
      std::string token, part, label;
      auto v = split(line, '\t');
      assert(v.size() == 3);
      token = v[0]; part = v[1]; label = v[2];
      if (w2i.find(token) == w2i.end()) {
        w2i[token] = i2w.size();
        i2w.push_back(token);
      }
      if (y2i.find(label) == y2i.end()) {
        y2i[label] = i2y.size();
        i2y.push_back(label);
      }
      x.push_back(w2i[token]);
      t.push_back(y2i[label]);
    }
  }
}

int main(int argc, char **argv) {
  int fold = -1;
  if (argc == 1) {
    std::cout << "Warning: No fold number as argument. Assuming 0."
              << std::endl;
    fold = 0;
  } else {
    fold = std::atoi(argv[1]); // between 0-9
  }
  srand(135);
  std::cout << std::setprecision(4);
  std::unordered_map<std::string,uint> w2i, y2i;
  std::vector<std::string> i2w, i2y;

  // start: read data

  std::string datadir = "../../data/mpqa/";

  std::vector<Data<cpu>> X, T;
  readSentences(X, T, w2i, i2w, y2i, i2y, datadir+"dse.txt"); // dse.txt or ese.txt

  std::unordered_map<std::string, std::set<uint> > sentenceIds;
  std::set<std::string> allDocs;
  std::ifstream in(datadir+"sentenceid.txt");
  std::string line;
  uint numericId = 0;
  while(std::getline(in, line)) {
    std::vector<std::string> s = split(line, ' ');
    assert(s.size() == 3);
    std::string strId = s[2];

    if (sentenceIds.find(strId) != sentenceIds.end()) {
      sentenceIds[strId].insert(numericId);
    } else {
      sentenceIds[strId] = std::set<uint>();
      sentenceIds[strId].insert(numericId);
    }
    numericId++;
  }

  std::vector<Data<cpu>> trainX, validX, testX;
  std::vector<Data<cpu>> trainL, validL, testL;
  std::vector<bool> isUsed(X.size(), false);

  std::ifstream in4(datadir+"datasplit/doclist.mpqaOriginalSubset");
  while(std::getline(in4, line))
    allDocs.insert(line);

  std::ifstream in2(datadir+"datasplit/filelist_train"+std::to_string(fold));
  while(std::getline(in2, line)) {
    for (const auto &id : sentenceIds[line]) {
      trainX.push_back(X[id]);
      trainL.push_back(T[id]);
    }
    allDocs.erase(line);
  }
  std::ifstream in3(datadir+"datasplit/filelist_test"+std::to_string(fold));
  while(std::getline(in3, line)) {
    for (const auto &id : sentenceIds[line]) {
      testX.push_back(X[id]);
      testL.push_back(T[id]);
    }
    allDocs.erase(line);
  }

  for (const auto &doc : allDocs) {
    for (const auto &id : sentenceIds[doc]) {
      validX.push_back(X[id]);
      validL.push_back(T[id]);
    }
  }

  std::cout << X.size() << " " << trainX.size()
            << " " << testX.size() << std::endl
            << "Valid size: " << validX.size() << std::endl;

  // end: read data

  // start: batch data

  std::vector<Data<gpu>> trainXb, validXb, testXb;
  std::vector<Data<gpu>> trainLb, validLb, testLb;

  uint batch_size = 32, nX = i2w.size(), nL = 3;
  batch_seq_label_seq(&trainXb, &trainLb, trainX, trainL, nX, nL, batch_size);
  batch_seq_label_seq(&validXb, &validLb, validX, validL, nX, nL, batch_size);
  batch_seq_label_seq(&testXb,  &testLb,  testX,  testL,  nX, nL, batch_size);

  trainX.clear(); trainL.clear(); // save host mem
  validX.clear(); validL.clear();
  testX.clear();  testL.clear();

  // end: batch data

  // start: network definition

  auto ds = datastream(2);
  auto wv = proj(300, nX+1);
  auto layer = []() { return cast() >>
                             (recurrent(100), recurrent(100, reverse)) >>
                             cat(); };
  auto top = smax_xent();
  auto nn = ds >> wv >> layer() >> layer() >> layer()
               >> ff(3,nonlin::id()) >> top;

  wv->init();
  load_wv_table("/home/oirsoy/glove.840B.300d.txt", 300, &(wv->W()), w2i);

  nn->set_updater<rmsprop<gpu>>();
  nn->set_lr(1e-3);
  nn->set_la(1e-4);
  wv->set_lr(0.); // fix word vec table, simpler.

  // end: network definition

  std::cout << "Vocab size: " << w2i.size() << std::endl;

  std::vector<Real> bestDevsTrain(6, 0),
                    bestDevsDev(6, 0),
                    bestDevsTest(6, 0); // for early stopping
  // {propP, propR, propF1, binP, binR, binF1}


  trainer<gpu> t(ds, nn);

  for (uint ep=0; ep <100; ep++) {   // training loop
    ds->set_data({&trainXb, &trainLb});
    timer::tic("train");
    Real running_err = t.train({&trainXb, &trainLb});
    timer::toc("train");

    if ((ep+1) % 5 == 0) {
      std::cout << "Epoch " << ep << std::endl;
      auto tra = Fscore<gpu>(nn, ds, top, i2y, trainXb, trainLb, nL);
      auto dev = Fscore<gpu>(nn, ds, top, i2y, validXb, validLb, nL);
      std::cout << "Train:" << std::endl;
      std::cout << tra << std::endl;
      std::cout << "Dev:" << std::endl;
      std::cout << dev << std::endl;
      if (dev[2] > bestDevsDev[2]) { // model selection is based on prop F1
        bestDevsTrain = tra;
        bestDevsDev = dev;
        bestDevsTest = Fscore<gpu>(nn, ds, top, i2y, testXb, testLb, nL);
      }
      std::cout << std::endl;
    }
  }
  std::cout << "Best:" << std::endl;
  std::cout << "Train:" << std::endl;
  std::cout << bestDevsTrain << std::endl;
  std::cout << "Dev:" << std::endl;
  std::cout << bestDevsDev << std::endl;
  std::cout << "Test:" << std::endl;
  std::cout << bestDevsTest << std::endl << std::endl;

  timer::print();

  return 0;
}
