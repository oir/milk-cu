#ifndef MILK_INIT_H
#define MILK_INIT_H

#include "base.h"

namespace milk {

namespace init {

static uint seed = 0.;

template <typename xpu>
void uniform(Matrix<xpu> W) {
  mshadow::Random<xpu, Real>(seed++).SampleUniform(&W, -0.01, 0.01);
}

} // end namespace init
} // end namespace milk

#endif
