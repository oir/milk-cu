#ifndef MILK_UTILS_IO_H
#define MILK_UTILS_IO_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace milk {

using namespace mshadow;
using namespace mshadow::expr;

bool is_whitespace(std::string s) {
  for (uint i=0; i<s.size(); i++) if (!isspace(s[i])) return false;
  return true;
}

std::vector<std::string> split(const std::string &s) { // splits from whitespace
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (ss >> item)
    elems.push_back(item);
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) { // splits from delim
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
  for (uint i=0; i<v.size(); i++) {
    s << v[i];
    if (i < v.size()-1) s << " ";
  }
  return s;
}

std::ostream& operator<<(std::ostream& s, const Vector<cpu>& v) {
  for (uint i=0; i<v.size(0); i++) {
    s << v[i];
    if (i < v.size(0)-1) s << " ";
  }
  return s;
}

std::ostream& operator<<(std::ostream& s, const Matrix<cpu>& m) {
  for (uint i=0; i<m.size(0); i++) {
    for (uint j=0; j<m.size(1); j++) {
      s << m[i][j];
      if (j < m.size(1)-1)
        s << " ";
    }
    if (i < m.size(0)-1)
      s << std::endl;
  }
  return s;
}

std::ostream& operator<<(std::ostream& s, const Vector<gpu>& v) {
  VectorContainer<cpu> v_(v.shape_);
  Copy(v_, v);
  s << v_;
  return s;
}

std::ostream& operator<<(std::ostream& s, const Matrix<gpu>& m) {
  MatrixContainer<cpu> m_(m.shape_);
  Copy(m_, m);
  s << m_;
  return s;
}

std::istream& operator>>(std::istream& s, milk::Matrix<cpu>& m) {
  for (uint i=0; i<m.size(0); i++)
    for (uint j=0; j<m.size(1); j++)
      s >> m[i][j];
  return s;
}

std::istream& operator>>(std::istream& s, milk::Matrix<gpu>& m) {
  MatrixContainer<cpu> m_(m.shape_);
  s >> m_;
  Copy(m, m_);
  return s;
}

void read_table(std::string fname, Matrix<cpu>* X) {
  // assume *X is allocated to proper size
  std::string line;
  std::ifstream in(fname.c_str());
  assert(in.is_open());

  for (uint i=0; i<X->size(0); i++) {
    std::getline(in, line);
    auto v = split(line);
    auto row = (*X)[i];
    for (uint j=0; j<X->size(1); j++) {
      row[j] = std::stod(v[j]);
    }
  }
}

void load_wv_table(std::string fname, uint d, Matrix<gpu>* W,
                   std::unordered_map<std::string,uint>& w2i) {
  std::string line;
  std::ifstream in(fname);
  assert(in.is_open());
  VectorContainer<cpu> tmp(Shape1(d));
  while (std::getline(in, line)) {
    auto v = split(line, ' ');
    std::string w = v[0];
    if (w2i.find(w) != w2i.end()) {
      uint ix = w2i[w];
      for (uint i=0; i<d; i++)
        tmp[i] = std::stod(v[i+1]);
      Copy((*W)[ix], tmp);
    }
  }
}

} // end namespace milk

#endif

