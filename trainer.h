#ifndef MILK_TRAINER_H
#define MILK_TRAINER_H

#include "base.h"

namespace milk {

template <typename xpu>
class trainer {
  public:
    // we want to be able to refer to ds separately for now
    std::shared_ptr<layer::datastream<xpu>> ds;
    std::shared_ptr<layer::layer<xpu>> all; // ds >> nn together

    trainer(std::shared_ptr<layer::datastream<xpu>> a_ds,
            std::shared_ptr<layer::layer<xpu>> a_all)
      : ds(a_ds), all(a_all) {}

    virtual Real run(std::vector<std::vector<Data<xpu>>*> dataset,
                     Mode mode, bool train, uint epoch=1,
                     uint max_len=std::numeric_limits<uint>::max(),
                     uint num_iter = 0) {
      all->set_mode(mode);
      ds->max_len = max_len;
      ds->set_data(dataset);
      Real err = 0.;
      uint tot = 0;
      for (uint e=0; e<epoch; e++) {
        do {
          all->forward();
          err += all->error();
          //tot += ds->x[0].batch_size;
          tot += ds->x[1]().size(0);  // TODO: maybe make denominator generic
          if (train) {
            all->backward();
            all->update();
          }
        } while (ds->count != num_iter);
      }
      return err/tot;
    }

    virtual Real train(std::vector<std::vector<Data<xpu>>*> dataset,
                       uint max_len=std::numeric_limits<uint>::max(),
                       uint epoch=1) {
      return run(dataset, TRAIN, true, epoch, max_len);
    }

    virtual Real mean_error(std::vector<std::vector<Data<xpu>>*> dataset,
                            uint max_len=std::numeric_limits<uint>::max()) {
      return run(dataset, TEST, false, 1, max_len);
    }
};

} // end namespace milk

#endif
