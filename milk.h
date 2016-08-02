#ifndef MILK_H
#define MILK_H

#define MSHADOW_USE_MKL 0
#define MSHADOW_USE_CBLAS true
#define MSHADOW_FORCE_STREAM 0
#define MSHADOW_ALLOC_PAD false
#include "mshadow/tensor.h"
#include "mshadow/random.h"
#include "mshadow/expression.h"
#include "mshadow/extension.h"

#include "defs.h"       // definitions and defaults
#include "base.h"       // Data, Input and Weights (NN stuff)
#include "utils/utils"  // useful small functions
#include "nonlin.h"     // NN nonlinearities (tanh, relu etc)
#include "layer/layer"  // all NN layers
#include "trainer.h"    // convenience functions for training NNs

#endif
