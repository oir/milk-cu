#ifndef MILK_UTILS_DAG_H
#define MILK_UTILS_DAG_H

namespace milk {

// (topologically) sorted DAG (for recursive nets for now)
// The assumption is that this is constructed in a way that is topologically
// sorted from root to leaves as the index is incremented. This is no way ensured
// by the class itself.

class sdag {
  public:
    // adj_list[n] lists children of node n as
    // (node index, incoming edge label) pairs
    std::vector<std::vector<std::pair<uint,uint>>> adj_list;

    std::vector<std::pair<uint,uint>>& children(uint n) { return adj_list[n]; }
    uint size() { return adj_list.size(); }
};

}; // end namespace milk

#endif
