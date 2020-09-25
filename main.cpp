#include <iostream>
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;

int main() {
    const int n = 1000, n_query = 10000;
    const string base_dir = "/home/arai/workspace/";
    const string data_path = base_dir + "dataset/deep/data10m/";
    const string query_path = base_dir + "dataset/deep/query10k.csv";
    const string graph_path = base_dir + "index/aknng/deep/data1m-d25/";
    const string ground_truth_path = base_dir + "result/knn-search/scan/deep/data1m/k100/result.csv";

    const auto queries = load_data(query_path, n_query);
    const auto ground_truth = load_neighbors(ground_truth_path, n_query, true);

    int m = 40;

    int k = 10;
    int l = k;

    auto index = NSG(m);
    index.build(data_path, graph_path, n);

    cout << "complete: build index" << endl;

    SearchResults results;
    for (const auto& query : queries) {
        auto result = index.knn_search(query, k, l);
        result.recall = calc_recall(result.result, ground_truth[query.id], k);
        results.push_back(move(result));
    }

    const string save_postfix = "k" + to_string(k) +
                                "-m" + to_string(m) +
                                "-l" + to_string(l) + ".csv";
    const string save_dir = base_dir + "result/knn-search/nsg/deep/data1m/k-vary/";
    const string log_path = save_dir + "log-" + save_postfix;
    const string result_path = save_dir + "result-" + save_postfix;
    results.save(log_path, result_path);
    }
}