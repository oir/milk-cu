// some useful operator overloads and other functions that did not fit
// anywhere else

#ifndef MILK_UTILS_OP_H
#define MILK_UTILS_OP_H

template <typename T>
std::vector<T> operator+(const std::vector<T>& l, const std::vector<T>& r) {
  std::vector<T> v(l);
  v.insert(v.end(), r.begin(), r.end());
  return v;
}

template <typename T>
std::vector<T*> get_ptrs(std::vector<T>& v) {
  std::vector<T*> v_;
  for (auto& x : v) v_.push_back(&x);
  return v_;
}

#endif
