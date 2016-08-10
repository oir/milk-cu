#ifndef MILK_UTILS_DATA_H
#define MILK_UTILS_DATA_H

#include <random>

namespace milk {

template <typename xpu>
void paired_shuffle(std::vector<Matrix<xpu>> v,
                    std::vector<uint> *perm = nullptr) {
  std::default_random_engine gen;

  if (perm) {
    *perm = std::vector<uint>(v[0].size(0));
    std::iota(perm->begin(), perm->end(), 0); // 0 .. N-1
  }
  for (uint i=v[0].size(0)-1; i>0; i--) {
    uint j = std::uniform_int_distribution<int>(0,i)(gen);
    for (auto m : v) {
      VectorContainer<xpu> tmp(m[0].shape_);
      Copy(tmp, m[i]);
      Copy(m[i], m[j]);
      Copy(m[j], tmp);
    }
    if (perm) std::swap((*perm)[i], (*perm)[j]);
  }
}

#if MSHADOW_USE_CUDA

std::vector<Data<gpu>> to_data(const Matrix<cpu>& X, uint batch_size=1) {
  uint N = X.size(0);
  uint last_batch_size = N % batch_size;
  uint num_batches = (N + batch_size - 1) / batch_size;
  if (last_batch_size == 0) last_batch_size = batch_size;

  std::vector<Data<gpu>> vdat;
  for (uint i=0; i<num_batches; i++) {
    uint bs = (i==(num_batches-1)) ? last_batch_size : batch_size;
    auto dat = Data<gpu>(bs, X.size(1));
    Copy(dat(), middle_rows(X, i*batch_size, bs));
    dat.batch_size = bs;
    vdat.push_back(dat);
  }
  return vdat;
}

void batch_seq_single_label(std::vector<Data<gpu>>* Xb,
                            std::vector<Data<gpu>>* Lb,
                            std::vector<Data<cpu>>& X,
                            std::vector<Data<cpu>>& L,
                            Real pad_value,
                            uint batch_size = 64,
                            bool from_right = false) {
  uint N = X.size();
  uint last_batch_size = N % batch_size;
  uint num_batches = (N + batch_size - 1) / batch_size;
  if (last_batch_size == 0) last_batch_size = batch_size;

  // sort wrt length
  std::vector<uint> index(X.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
      [&](const int& a, const int& b) {
        return (X[a]().shape_[0] > X[b]().shape_[0]);
      }
  );

  Xb->resize(num_batches);
  Lb->resize(num_batches);

  for (uint i=0; i<num_batches; i++) {
    uint bs = (i == (num_batches-1)) ? last_batch_size : batch_size;
    uint T = X[index[i*bs]]().size(0); // max length in batch
    (*Xb)[i].init(bs*T, X[index[i*bs]]().size(1));
    (*Lb)[i].init(bs,   L[index[i*bs]]().size(1));

    for (uint j=0; j<bs; j++) {
      uint T_ = X[index[i*bs + j]]().size(0);
      for (uint t=0; t<T; t++) {
        if (from_right) {
          if (t < T_) {
            Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t] );
          } else { // pad here
            (*Xb)[i]()[bs*t + j] = pad_value;
          }
        } else {
          if (t >= T-T_) {
            Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t-(T-T_)] );
          } else { // or here
            (*Xb)[i]()[bs*t + j] = pad_value;
          }
        }
      }
      Copy( (*Lb)[i]()[j], L[index[i*bs + j]]()[0] );
    }
    (*Xb)[i].batch_size = (*Lb)[i].batch_size = bs;
  }
}

// TODO: test left / right padding
void batch_seq_label_seq(std::vector<Data<gpu>>* Xb,
                         std::vector<Data<gpu>>* Lb,
                         std::vector<Data<cpu>>& X,
                         std::vector<Data<cpu>>& L,
                         Real pad_value,
                         Real label_pad_value,
                         uint batch_size = 64,
                         bool from_right = false) {
  uint N = X.size();
  uint last_batch_size = N % batch_size;
  uint num_batches = (N + batch_size - 1) / batch_size;
  if (last_batch_size == 0) last_batch_size = batch_size;

  // sort wrt length
  std::vector<uint> index(X.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
      [&](const int& a, const int& b) {
        return (X[a]().shape_[0] > X[b]().shape_[0]);
      }
  );

  Xb->resize(num_batches);
  Lb->resize(num_batches);

  for (uint i=0; i<num_batches; i++) {
    uint bs = (i == (num_batches-1)) ? last_batch_size : batch_size;
    uint T = X[index[i*bs]]().size(0); // max length in batch
    (*Xb)[i].init(bs*T, X[index[i*bs]]().size(1));
    (*Lb)[i].init(bs*T, L[index[i*bs]]().size(1));

    for (uint j=0; j<bs; j++) {
      uint T_ = X[index[i*bs + j]]().size(0);
      for (uint t=0; t<T; t++) {
        if (from_right) {
          if (t < T_) {
            Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t] );
            Copy( (*Lb)[i]()[bs*t + j], L[index[i*bs + j]]()[t] );
          } else { // pad here
            (*Xb)[i]()[bs*t+j] = pad_value;
            (*Lb)[i]()[bs*t+j] = label_pad_value;
          }
        } else {
          if (t >= T-T_) {
            Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t-(T-T_)] );
            Copy( (*Lb)[i]()[bs*t + j], L[index[i*bs + j]]()[t-(T-T_)] );
          } else { // or here
            (*Xb)[i]()[bs*t+j] = pad_value;
            (*Lb)[i]()[bs*t+j] = label_pad_value;
          }
        }
      }
    }
    (*Xb)[i].batch_size = (*Lb)[i].batch_size = bs;
  }
}

void batch_seq_no_label(std::vector<Data<gpu>>* Xb,
                        std::vector<Data<cpu>>& X,
                        Real pad_value,
                        uint batch_size = 64,
                        bool from_right = false) {
  uint N = X.size();
  uint last_batch_size = N % batch_size;
  uint num_batches = (N + batch_size - 1) / batch_size;
  if (last_batch_size == 0) last_batch_size = batch_size;

  // sort wrt length
  std::vector<uint> index(X.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
      [&](const int& a, const int& b) {
        return (X[a]().shape_[0] > X[b]().shape_[0]);
      }
  );

  Xb->resize(num_batches);

  for (uint i=0; i<num_batches; i++) {
    uint bs = (i == (num_batches-1)) ? last_batch_size : batch_size;
    uint T = X[index[i*bs]]().size(0); // max length in batch
    (*Xb)[i].init(bs*T, X[index[i*bs]]().size(1));
    MatrixContainer<cpu> tmp((*Xb)[i]().shape_);

    for (uint j=0; j<bs; j++) {
      uint T_ = X[index[i*bs + j]]().size(0);
      for (uint t=0; t<T; t++) {
        if (from_right) {
          if (t < T_) {
            //Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t] );
            Copy( tmp[bs*t + j], X[index[i*bs + j]]()[t] );
          } else { // pad here
            //(*Xb)[i]()[t] = pad_value;
            tmp[bs*t + j] = pad_value;
          }
        } else {
          if (t >= T-T_) {
            //Copy( (*Xb)[i]()[bs*t + j], X[index[i*bs + j]]()[t-(T-T_)] );
            Copy( tmp[bs*t + j], X[index[i*bs + j]]()[t-(T-T_)] );
          } else { // or here
            //(*Xb)[i]()[t] = pad_value;
            tmp[bs*t + j] = pad_value;
          }
        }
      }
    }
    Copy( (*Xb)[i](), tmp );
    (*Xb)[i].batch_size = bs;
  }
}

#endif

} // end namespace milk

#endif

