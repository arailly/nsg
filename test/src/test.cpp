#include "gtest/gtest.h"
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;

TEST(graph, knn_search_with_checked) {
    const auto p0 = Data<>(0, {1});
    const auto p1 = Data<>(1, {2});
    const auto p2 = Data<>(2, {3});
    const auto p3 = Data<>(3, {4});
    const auto p4 = Data<>(4, {5});
    const auto p5 = Data<>(5, {6});

    auto series = Dataset<>{p0, p1, p2, p3, p4, p5};
    auto nsg = NSG(5, 3);
    nsg.init_nodes(series);
    nsg.navi_node_id = 0;
    nsg.nodes[0].add_neighbor(1, 1);
    nsg.nodes[1].add_neighbor(1, 2);
    nsg.nodes[2].add_neighbor(1, 3);
    nsg.nodes[3].add_neighbor(1, 4);
    nsg.nodes[4].add_neighbor(1, 5);
    nsg.nodes[5].add_neighbor(1, 4);

    const auto& query_node = nsg.nodes[5];

    const auto result = nsg.calc_neighbor_candidates(query_node);
    ASSERT_EQ(result[0].id, 4);
    ASSERT_EQ(result[1].id, 3);
    ASSERT_EQ(result[2].id, 2);
    ASSERT_EQ(result[3].id, 1);
    ASSERT_EQ(result[4].id, 0);
}

//TEST(nsg, find_navi_node_id) {
//    const auto p0 = Point(0, {1});
//    const auto p1 = Point(1, {2});
//    const auto p2 = Point(2, {3});
//    const auto p3 = Point(3, {4});
//    const auto p4 = Point(4, {5});
//
//    const uint n_sample = 5;
//
//    auto series = Series{p0, p1, p2, p3, p4};
//    auto nsg = NSG();
//    nsg.init_nodes(series);
//
//    const auto navi_node_id = nsg.find_navi_node_id(n_sample);
//    ASSERT_EQ(navi_node_id, 2);
//}
//
//TEST(nsg, conflict) {
//    const auto p0 = Point(0, {0, 0});
//    const auto p1 = Point(1, {1, 3});
//    const auto p2 = Point(2, {3, 2});
//    const auto p3 = Point(3, {1, 2});
//
//    auto series = Series{p0, p1, p2, p3};
//    auto series_copy = series;
//
//    auto nsg1 = NSG();
//    nsg1.init_nodes(series);
//    nsg1.nodes[0].add_neighbor(nsg1.nodes[1]);
//    nsg1.nodes[1].add_neighbor(nsg1.nodes[0]);
//    nsg1.nodes[1].add_neighbor(nsg1.nodes[2]);
//    nsg1.nodes[2].add_neighbor(nsg1.nodes[1]);
//
//    ASSERT_TRUE(nsg1.conflict(nsg1.nodes[0], nsg1.nodes[2]));
//
//    auto nsg2 = NSG();
//    nsg2.init_nodes(series_copy);
//    nsg2.nodes[0].add_neighbor(nsg2.nodes[2]);
//    nsg2.nodes[2].add_neighbor(nsg2.nodes[0]);
//    nsg2.nodes[2].add_neighbor(nsg2.nodes[3]);
//    nsg2.nodes[3].add_neighbor(nsg2.nodes[2]);
//
//    ASSERT_FALSE(nsg2.conflict(nsg2.nodes[0], nsg2.nodes[3]));
//}
//
//TEST(nsg, create_nsg) {
//    string data_path = "./data.csv";
//    string knng_path = "./knng.csv";
//    uint n = 8, m = 4, n_sample = 8, l = 4, c = 8;
//    auto nsg = NSG();
//    nsg.build(data_path, knng_path, m, n_sample, n, l, c);
//
//    ASSERT_EQ(nsg.nodes[0].neighbors[0].get().point.id, 1);
//    ASSERT_EQ(nsg.nodes[0].neighbors[1].get().point.id, 2);
//    ASSERT_EQ(nsg.nodes[0].neighbors[2].get().point.id, 4);
//    ASSERT_EQ(nsg.nodes[0].neighbors[3].get().point.id, 3);
//
//    ASSERT_EQ(nsg.nodes[4].neighbors[0].get().point.id, 1);
//    ASSERT_EQ(nsg.nodes[4].neighbors[1].get().point.id, 2);
//    ASSERT_EQ(nsg.nodes[4].neighbors[2].get().point.id, 6);
//    ASSERT_EQ(nsg.nodes[4].neighbors[3].get().point.id, 7);
//}
//
//TEST(nsg, search) {
//    unsigned n = 10, k = 10, m = 30, n_query = 20, n_sample = 1000, l = 40, c = 500;
//    string data_dir = "/home/arai/workspace/dataset/sift/sift_base/";
//    string query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";
//    string knng_path = "/home/arai/workspace/index/aknng/sift/sift10k-k20.csv";
//    auto nsg = NSG();
//    nsg.build(data_dir, knng_path, m, n_sample, n, l, c);
//
//    const auto series = load_data(data_dir, n);
//    const auto queries = read_csv(query_path, n_query);
//
//    for (const auto& query : queries) {
//        // calculate exact result by scan
//        const auto scan_result = scan_knn_search(query, k, series);
//
//        // calculate result by nsg
//        const auto nsg_result = nsg.knn_search(query, k, 40);
//        const auto& result = nsg_result.result;
//
//        ASSERT_EQ(result.size(), k);
//        float recall = 0;
//        for (const auto& exact : scan_result) {
//            for (const auto& approx : result) {
//                recall += (exact.id == approx.id);
//            }
//        }
//        recall /= k;
//
//        ASSERT_GE(recall, 0);
//    }
//}
//
//TEST(nsg, angular) {
//    unsigned n = 1000, k = 10, m = 30, n_query = 20, n_sample = 1000, l = 40, c = 500;
//    string data_dir = "/Users/yusuke-arai/workspace/dataset/glove/twitter/d100/data1m/";
//    string query_path = "/Users/yusuke-arai/workspace/dataset/glove/twitter/d100/query10k.csv";
//    string knng_path = "/Users/yusuke-arai/workspace/index/glove1m-k50/";
//    const auto series = load_data(data_dir, n);
//    auto series_copy = series;
//
//    auto nsg = NSG("angular");
//    nsg.build(data_dir, knng_path, m, n_sample, n, l, c);
//
//    const auto queries = read_csv(query_path, n_query);
//
//    for (const auto& query : queries) {
//        // calculate exact result by scan
//        const auto scan_result = scan_knn_search(query, k, series);
//
//        // calculate result by nsg
//        const auto result = nsg.knn_search(query, k, 40).result;
//
//        ASSERT_EQ(result.size(), k);
//        float recall = 0;
//        for (const auto& exact : scan_result) {
//            for (const auto& approx : result) {
//                recall += (exact.id == approx.id);
//            }
//        }
//        recall /= k;
//
//        ASSERT_GE(recall, 0);
//    }
//}
//
//TEST(nsg, knn_search_result) {
//    unsigned n = 1000, k = 5, n_query = 10000, l = 30;
//    string data_path = "/home/arai/workspace/dataset/sift/sift_base/";
//    string query_path = "/home/arai/workspace/dataset/sift/sift_query.csv";
//    string nsg_path = "/home/arai/workspace/index/nsg/sift/data1m-m50/";
//
//    auto nsg = NSG();
//    nsg.load(data_path, nsg_path, n);
//
//    const auto queries = read_csv(query_path, n_query);
//
//    SearchResults results;
//    for (const auto& query : queries) {
//        const auto result = nsg.knn_search(query, k, l);
//        results.push_back(result);
//    }
//
//    const string log_path = "/home/arai/workspace/result/knn-search/nsg/sift/data1m/k5/"
//                            "log-m50-l40.csv";
//    const string result_path = "/home/arai/workspace/result/knn-search/nsg/sift/data1m/k5/"
//                               "result-m50-l40.csv";
//    results.save(log_path, result_path);
//}