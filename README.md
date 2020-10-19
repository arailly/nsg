# NSG
## Overview
It is implementation of approximate kNN search on NSG: Navigating Spreading-out Graph.

The main algorithm is written `include/nsg.hpp` (header only), so you can use it easily.

Reference: Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph (C. Fu et al., VLDB 2019)

## Example
```
#include <iostream>
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;
using namespace nsg;

int main() {
    const string data_path = "path/to/data.csv";
    const string aknng_path = "path/to/aknng.csv"; // path for AKNNG (for pre-index)
    const string query_path = "path/to/query.csv";
    const string save_path = "path/to/result.csv";
    const unsigned n = 1000; // data size
    const unsigned n_query = 10; // query size
    const int k = 10; // result size
    const int l = 15; // candidate set size
    const int m = 30; // degree

    const auto queries = read_csv(query_path, n_query); // read query file

    auto nsg = NSG(m); // init NSG index
    hnsw.build(data_path, aknng_path, n); // build index
    
    vector<SearchResult> results(queries.size());
    for (const auto& query : queries) {
        results[query.id] = nsg.knn_search(query, k, l);
    }
}
```

## Input File Format
If you want to create index with this three vectors, `(0, 1), (2, 4), (3, 3)`, you must describe data.csv like following format:
```
0,1
2,4
3,3
```

Query format is same as data format.
