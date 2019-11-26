#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <deque>
#include "gtest/gtest.h"
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;

Series scan_knn_search(const Point& query, const uint k, const Series& series) {
    multimap<float, Point> result_map;
    for (unsigned i = 0; i < series.size(); i++) {
        const auto& point = series[i];
        const auto d = euclidean_distance(query, point);
        result_map.emplace(d, point);
    }

    auto result = Series();
    for (const auto& result_pair : result_map) {
        result.push_back(result_pair.second);
        if (result.size() >= k) break;
    }
    return result;
}

TEST(graph, knn_search_with_checked) {
    const auto query = Point(5, {6});
    const auto k = 3;

    const auto p0 = Point(0, {1});
    const auto p1 = Point(1, {2});
    const auto p2 = Point(2, {3});
    const auto p3 = Point(3, {4});
    const auto p4 = Point(4, {5});

    auto series = Series{p0, p1, p2, p3, p4};
    auto graph = Graph(series);
    graph[0].add_neighbor(graph[1]);
    graph[1].add_neighbor(graph[2]);
    graph[2].add_neighbor(graph[3]);
    graph[3].add_neighbor(graph[4]);

    const auto result = knn_search_with_checked(query, k, graph[0]);
    ASSERT_EQ(result[0].get().point.id, 4);
    ASSERT_EQ(result[1].get().point.id, 3);
    ASSERT_EQ(result[2].get().point.id, 2);
    ASSERT_EQ(result[3].get().point.id, 1);
    ASSERT_EQ(result[4].get().point.id, 0);
}

TEST(nsg, find_navi_node_id) {
    const auto p0 = Point(0, {1});
    const auto p1 = Point(1, {2});
    const auto p2 = Point(2, {3});
    const auto p3 = Point(3, {4});
    const auto p4 = Point(4, {5});

    const uint n_sample = 5;

    auto series = Series{p0, p1, p2, p3, p4};
    auto graph = Graph(series);

    const auto navi_node_id = find_navi_node_id(graph, n_sample);
    ASSERT_EQ(navi_node_id, 2);
}

TEST(nsg, conflict) {
    const auto p0 = Point(0, {0, 0});
    const auto p1 = Point(1, {1, 3});
    const auto p2 = Point(2, {3, 2});
    const auto p3 = Point(3, {1, 2});

    auto series = Series{p0, p1, p2, p3};
    auto graph1 = Graph(series);
    graph1[0].add_neighbor(graph1[1]);
    graph1[1].add_neighbor(graph1[0]);
    graph1[1].add_neighbor(graph1[2]);
    graph1[2].add_neighbor(graph1[1]);

    ASSERT_TRUE(conflict(graph1[0], graph1[2]));

    auto graph2 = Graph(series);
    graph2[0].add_neighbor(graph1[2]);
    graph2[2].add_neighbor(graph1[0]);
    graph2[2].add_neighbor(graph1[3]);
    graph2[3].add_neighbor(graph1[2]);

    ASSERT_FALSE(conflict(graph2[0], graph2[3]));
}

//TEST(nsg, create_nsg) {
//    string data_path = "./data.csv";
//    string knng_path = "./knng.csv";
//    uint m = 4;
//    uint n_sample = 8;
//    const auto nsg = create_nsg(data_path, knng_path, m, m, n_sample, n_sample);
//
//    ASSERT_EQ(nsg[0].neighbors[0].get().point.id, 1);
//    ASSERT_EQ(nsg[0].neighbors[1].get().point.id, 2);
//    ASSERT_EQ(nsg[0].neighbors[2].get().point.id, 4);
//    ASSERT_EQ(nsg[0].neighbors[3].get().point.id, 3);
//
//    ASSERT_EQ(nsg[4].neighbors[0].get().point.id, 0);
//    ASSERT_EQ(nsg[4].neighbors[1].get().point.id, 1);
//    ASSERT_EQ(nsg[4].neighbors[2].get().point.id, 2);
//    ASSERT_EQ(nsg[4].neighbors[3].get().point.id, 6);
//}

TEST(nsg, search) {
    unsigned nk = 2, k = 5, m = 50, n_query = 20, n_sample = 1000;
    string data_dir = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    string knng_path = "/Users/yusuke-arai/workspace/index/sift1k-k20.csv";
    const auto nsg = create_nsg(data_dir, knng_path, m, k, n_sample, nk);

    const auto [series, queries] = [&data_dir, nk, n_query]() -> pair<Series, Series> {
        const auto series_with_query = load_data(data_dir, nk + n_query);
        Series ss, qs;
        for (int i = 0; i < nk * 1000; i++) ss.push_back(series_with_query[i]);
        for (int i = nk * 1000; i < nk * 1000 + n_query; i++) qs.push_back(series_with_query[i]);
        return make_pair(ss, qs);
    }();

    for (const auto& query : queries) {
        // calcurate exact result by scan
        const auto scan_result = scan_knn_search(query, k, series);

        // calcurate result by nsg
        const auto result = knn_search(query, k, nsg.navi_node);

        ASSERT_EQ(result.size(), k);
        float recall = 0;
        for (const auto& exact : scan_result) {
            for (const auto& approx : result) {
                recall += (exact.id == approx.get().point.id);
            }
        }
        recall /= k;

        ASSERT_GE(recall, 0);
    }
}

TEST(nsg, knn_search) {
    unsigned k = 10;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    const string query_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_query.csv";
    const string nsg_path = "/Users/yusuke-arai/workspace/index/nsg-sift1m-m50.csv";
    const auto nsg = load_nsg(data_path, nsg_path);
}

TEST(s, s) {
    unsigned m = 40, k = 20, n = 10000, n_sample = 1000;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    const string knng_path = "/Users/yusuke-arai/workspace/index/sift10k-k20.csv";
    const string save_path = "/Users/yusuke-arai/workspace/index/nsg-sift10k-m40.csv";
    auto nsg = create_nsg(data_path, knng_path, m, k, n_sample, n);
}