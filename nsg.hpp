//
// Created by Yusuke Arai on 2019/11/08.
//

#ifndef NSG_NSG_HPP
#define NSG_NSG_HPP

#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <deque>
#include <string>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>
#include <arailib.hpp>
#include <graph.hpp>

using namespace std;
using namespace arailib;
using namespace graph;

struct NSG {
    vector<Node> nodes;
    Node* navi_node;
    DistanceFunction df;
    mt19937 engine;

    NSG(const string& df = "euclidean", unsigned random_state = 42) :
        df(select_distance(df)), engine(mt19937(random_state)) {}

    void init_nodes(Series& series) {
        for (auto& point : series) nodes.emplace_back(point);
    }

    void load(Series& series, const string& graph_path, int n) {
        init_nodes(series);
        // csv
        if (is_csv(graph_path)) {
            ifstream ifs(graph_path);
            if (!ifs) throw runtime_error("Can't open file!");

            string line; getline(ifs, line);
            unsigned navi_node_id = stoi(line);
            navi_node = &nodes[navi_node_id];

            while (getline(ifs, line)) {
                const auto&& ids = split<size_t>(line);
                nodes[ids[0]].add_neighbor(nodes[ids[1]]);
            }
            return;
        }

        // dir
        const string navi_node_path = graph_path + "/navi-node.csv";
        ifstream navi_node_ifs(navi_node_path);
        if (!navi_node_ifs) throw runtime_error("Can't open file!");

        string navi_node_line; getline(navi_node_ifs, navi_node_line);
        unsigned navi_node_id = stoi(navi_node_line);

        navi_node = &nodes[navi_node_id];

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const string path = graph_path + "/" + to_string(i) + ".csv";
            ifstream ifs(path);
            string line;
            while (getline(ifs, line)) {
                const auto ids = split<size_t>(line);
                nodes[ids[0]].add_neighbor(nodes[ids[1]]);
            }
        }
    }

    void load(const string& data_path, const string& graph_path, int n) {
        // load data
        auto series = [&data_path, n]() {
            // csv
            if (is_csv(data_path)) return read_csv(data_path, n);
            // dir
            return load_data(data_path, n);
        }();

        init_nodes(series);
        load(series, graph_path, n);
    }

    void save(const string& save_dir) {
        const string navi_node_path = save_dir + "/navi-node.csv";
        ofstream navi_node_ofs(navi_node_path);
        // write navigating node
        navi_node_ofs << navi_node->point.id << endl;

        // write connection
        vector<string> lines(ceil(nodes.size() / 1000.0));
        for (const auto& node : nodes) {
            const unsigned line_i = node.point.id / 1000;
            for (const auto& neighbor : node.neighbors) {
                lines[line_i] += to_string(node.point.id) + ',' +
                                 to_string(neighbor.get().point.id) + '\n';
            }
        }

        for (int i = 0; i < lines.size(); i++) {
            const string path = save_dir + "/" + to_string(i) + ".csv";
            ofstream ofs(path);
            ofs << lines[i];
        }
    }

    vector<reference_wrapper<const Node>>
    knn_search(const Point query, const unsigned k, const unsigned l) {
        unordered_map<size_t, bool> checked, added;
        added[navi_node->point.id] = true;

        multimap<float, reference_wrapper<const Node>> candidates;
        const auto dist_to_start_node = df(query, navi_node->point);
        candidates.emplace(dist_to_start_node, *navi_node);

        while (true) {
            // find the first unchecked node
            const auto& first_unchecked_pair_ptr = [&candidates, &checked]() {
                auto candidate_pair_ptr = candidates.begin();
                for (;candidate_pair_ptr != candidates.end(); ++candidate_pair_ptr) {
                    const auto &candidate = candidate_pair_ptr->second.get();

                    if (checked[candidate.point.id]) continue;
                    checked[candidate.point.id] = true;

                    break;
                }
                return candidate_pair_ptr;
            }();

            if (distance(candidates.begin(), first_unchecked_pair_ptr) >= l) break;

            const auto& first_unchecked_node = first_unchecked_pair_ptr->second.get();

            for (const auto& neighbor : first_unchecked_node.neighbors) {
                if (added[neighbor.get().point.id]) continue;
                added[neighbor.get().point.id] = true;

                const auto dist = df(query, neighbor.get().point);
                candidates.emplace(dist, neighbor.get());
            }

            // resize candidates l
            while (candidates.size() > l) candidates.erase(--candidates.cend());
        }

        vector<reference_wrapper<const Node>> result;
        for (const auto& c : candidates) {
            result.emplace_back(c.second.get());
            if (result.size() >= k) break;
        }
        return result;
    }

    vector<reference_wrapper<const Node>>
    knn_search_with_checked(const Node& query_node, const unsigned l = 40,
                            const unsigned c = 500) {

        unordered_map<size_t, bool> checked, added;
        added[navi_node->point.id] = true;

        multimap<float, reference_wrapper<const Node>> candidates, checked_nodes;
        const auto dist_to_start_node = df(query_node.point, navi_node->point);
        candidates.emplace(dist_to_start_node, *navi_node);
        checked_nodes.emplace(dist_to_start_node, *navi_node);

        while (true) {
            // find the first unchecked node
            const auto& first_unchecked_pair_ptr = [&]() {
                auto candidate_pair_ptr = candidates.begin();
                for (;candidate_pair_ptr != candidates.end(); ++candidate_pair_ptr) {
                    const auto &candidate = candidate_pair_ptr->second.get();

                    if (checked[candidate.point.id]) continue;
                    checked[candidate.point.id] = true;

                    break;
                }
                return candidate_pair_ptr;
            }();

            if (distance(candidates.begin(), first_unchecked_pair_ptr) >= l ||
                first_unchecked_pair_ptr == candidates.end()) break;

            const auto& first_unchecked_node = first_unchecked_pair_ptr->second.get();

            for (const auto& neighbor : first_unchecked_node.neighbors) {
                if (added[neighbor.get().point.id]) continue;
                added[neighbor.get().point.id] = true;

                const auto dist = df(query_node.point, neighbor.get().point);
                candidates.emplace(dist, neighbor.get());
                checked_nodes.emplace(dist, neighbor.get());
            }

            auto a = [&]() {
                vector<Node> v;
                for (const auto& c : candidates) v.push_back(c.second);
                return v;

            }();
            auto b = 1;

            // resize candidates l
            while (candidates.size() > l) candidates.erase(--candidates.cend());
        }

        // add query_node's kNN
        for (const auto& neighbor : query_node.neighbors) {
            if (checked[neighbor.get().point.id]) continue;
            checked[neighbor.get().point.id] = true;
            const auto d = euclidean_distance(query_node.point, neighbor.get().point);
            checked_nodes.emplace(d, neighbor.get());
        }

        // add nodes into vector
        vector<reference_wrapper<const Node>> result;
        for (const auto& node : checked_nodes) {
            if (node.second.get().point == query_node.point) continue;
            result.emplace_back(node.second.get());
            if (result.size() >= c) break;
        }

        return result;
    }

    size_t find_navi_node_id(uint n_sample) {
        uniform_int_distribution<size_t> distribution(0, nodes.size() - 1);

        // sampling
        const auto sampled = [&]() mutable {
            vector<reference_wrapper<const Node>> v;
            unordered_map<size_t, bool> added;
            for (uint i = 0; i < n_sample; i++) {
                auto random_id = distribution(engine);
                if (added[random_id]) {
                    i--;
                    continue;
                }
                added[random_id] = true;
                v.emplace_back(nodes[random_id]);
            }
            return v;
        }();

        // calc sum of df of each points
        const auto distance_list = [&]() {
            vector<float> dl(n_sample);
#pragma omp parallel for
            for (unsigned i = 0; i < n_sample; i++) {
                const auto& node1 = sampled[i];
                float distance_sum = 0;
                for (const auto& node2 : sampled) {
                    if (node1.get().point == node2.get().point) continue;
                    const auto dist = df(node1.get().point, node2.get().point);
                    distance_sum += dist;
                }
                dl[i] = distance_sum;
            }
            return dl;
        }();

        // find argmin and return
        return [&distance_list, n_sample]() {
            size_t argmin = 0;
            float min_distance = distance_list[argmin];
            for (int i = 1; i < n_sample; i++) {
                auto d = distance_list[i];
                if (min_distance > d) {
                    min_distance = d;
                    argmin = i;
                }
            }
            return argmin;
        }();
    }

    bool conflict(const Node& v, const Node& p) const {
        // true if p is in v's neighbor r's neighbor (edge pr is not detour)
        for (const auto& r : v.neighbors) {
            if (r.get().point.id == p.point.id) return true;
            for (const auto& r_neighbor : r.get().neighbors) {
                if (r_neighbor.get().point.id != p.point.id) continue;

                // check if pr is not detour
                auto dist_from_v_to_p = df(p.point, v.point);
                auto dist_from_v_to_r = df(r.get().point, v.point);
                auto dist_from_r_to_p = df(p.point, r.get().point);
                if (dist_from_v_to_r < dist_from_v_to_p &&
                        dist_from_v_to_p > dist_from_r_to_p) return true;
            }
        }
        return false;
    }

    void dfs(const Node& node, unordered_map<size_t, bool>& visited) {
        for (const auto& neighbor : node.neighbors) {
            if (visited[neighbor.get().point.id]) continue;
            visited[neighbor.get().point.id] = true;
            dfs(neighbor.get(), visited);
        }
    }

    void build(const string& data_path, const string& knng_path, unsigned m,
               unsigned n_sample, int n, int l, int c) {
        // init
        auto series_for_knng = load_data(data_path, n);
        auto series_for_nsg = series_for_knng;

        const auto& knn_graph = load_graph(series_for_knng, knng_path, n);
        init_nodes(series_for_nsg);

        const auto navi_node_id = find_navi_node_id(n_sample);
        const auto& navi_node_knng = knn_graph[navi_node_id];
        navi_node = &nodes[navi_node_id];

        cout << "complete: load data and kNNG" << endl;

        // create
        const auto checked_node_list_along_search = [&]() {
            vector<vector<reference_wrapper<const Node>>> node_list(knn_graph.size());
#pragma omp parallel for
            for (unsigned i = 0; i < knn_graph.size(); i++) {
                const auto& v = knn_graph[i];
                const auto nodes = knn_search_with_checked(v, l, c);
                node_list[i] = nodes;
            }
            return node_list;
        }();

        cout << "complete: calcurate kNN" << endl;

        const unsigned par = nodes.size() / 10;
        for (unsigned i = 0; i < nodes.size(); i++) {
            auto& v = nodes[i];

            // show progress
            if (v.point.id % par == 0) {
                float progress = v.point.id / par * 10;
                cout << "progress: " << progress << "%" << endl;
            }

            for (const auto& p_knng : checked_node_list_along_search[i]) {
                auto& p = nodes[p_knng.get().point.id];
                if (p.point == v.point && conflict(v, p)) continue;
                v.add_neighbor(p);
                if (v.get_n_neighbors() >= m) break;
            }
        }

        cout << "complete: create NSG" << endl;

        while (true) {
            // check connection with dfs
            auto connected = [&]() {
                auto visited = unordered_map<size_t, bool>(nodes.size());
                visited[navi_node_id] = true;
                dfs(*navi_node, visited);
                return visited;
            }();

            // connect disconnected node
            bool all_connected = true;
            for (const auto& node : nodes) {
                if (connected[node.point.id]) continue;
                auto& disconnected_node = nodes[node.point.id];
                const auto knn_to_disconnected = knn_search(disconnected_node.point, 1, l);
                auto& nn_to_disconnected = nodes[knn_to_disconnected[0].get().point.id];
                nn_to_disconnected.add_neighbor(disconnected_node);
                all_connected = false;
            }
            if (all_connected) break;
        }

        cout << "complete: check connection" << endl;
    }
};


#endif //NSG_NSG_HPP
