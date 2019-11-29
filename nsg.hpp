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
#include <arailib.hpp>

namespace arailib {
    struct NSG : public Graph {
        Node& navi_node;
        NSG(Series& series, size_t id) : Graph(series), navi_node(nodes[id]) {}
    };

    NSG load_nsg(Series& series, const string& graph_path, int n) {
        // csv
        if (is_csv(graph_path)) {
            ifstream ifs(graph_path);
            if (!ifs) throw runtime_error("Can't open file!");

            string line; getline(ifs, line);
            unsigned navi_node_id = stoi(line);

            NSG nsg(series, navi_node_id);
            while (getline(ifs, line)) {
                const auto&& ids = split<size_t>(line);
                nsg[ids[0]].add_neighbor(nsg[ids[1]]);
            }
            return nsg;
        }

        // dir
        const string navi_node_path = graph_path + "/navi-node.csv";
        ifstream navi_node_ifs(navi_node_path);
        if (!navi_node_ifs) throw runtime_error("Can't open file!");

        string navi_node_line; getline(navi_node_ifs, navi_node_line);
        unsigned navi_node_id = stoi(navi_node_line);

        NSG nsg(series, navi_node_id);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const string path = graph_path + "/" + to_string(i) + ".csv";
            ifstream ifs(path);
            string line;
            while (getline(ifs, line)) {
                const auto ids = split<size_t>(line);
                nsg[ids[0]].add_neighbor(nsg[ids[1]]);
            }
        }
        return nsg;
    }

    NSG load_nsg(const string& data_path, const string& graph_path, int n) {
        // load data
        auto series = [&data_path, n]() {
            // csv
            if (is_csv(data_path)) return read_csv(data_path, n);
            // dir
            return load_data(data_path, n);
        }();

        // load nsg
        return load_nsg(series, graph_path, n);
    }

    void write_graph(const string& save_dir, const NSG& nsg) {
        const string navi_node_path = save_dir + "/navi-node.csv";
        ofstream navi_node_ofs(navi_node_path);
        // write navigating node
        navi_node_ofs << nsg.navi_node.point.id << endl;

        // write connection
        vector<string> lines(ceil(nsg.size() / 1000.0));
        for (const auto& node : nsg) {
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

    // nsw's knn_search with visited node
    vector<reference_wrapper<const Node>>
    knn_search_with_checked(const Node& query_node, const uint k, const Node& start_node) {
        unordered_map<size_t, bool> checked;
        checked[start_node.point.id] = true;

        multimap<float, reference_wrapper<const Node>> candidates, result_map, checked_nodes;
        const auto distance_to_start_node = euclidean_distance(query_node.point, start_node.point);
        candidates.emplace(distance_to_start_node, start_node);
        result_map.emplace(distance_to_start_node, start_node);
        checked_nodes.emplace(distance_to_start_node, start_node);

        while (!candidates.empty()) {
            const auto nearest_pair = candidates.extract(candidates.begin());
            const auto& distance_to_nearest = nearest_pair.key();
            const Node& nearest = nearest_pair.mapped().get();

            auto& furthest = *(--result_map.cend());
            // check if all elements are evaluated
            if (distance_to_nearest > furthest.first) break;

            for (const auto& neighbor : nearest.neighbors) {
                if (checked[neighbor.get().point.id]) continue;
                checked[neighbor.get().point.id] = true;

                const auto& distance_to_neighbor = euclidean_distance(query_node.point, neighbor.get().point);

                checked_nodes.emplace(distance_to_neighbor, neighbor.get());

                if (result_map.size() < k) {
                    candidates.emplace(distance_to_neighbor, neighbor.get());
                    result_map.emplace(distance_to_neighbor, neighbor.get());
                    continue;
                }

                const auto& furthest_ = *(--result_map.end());
                const auto& distance_to_furthest_ = euclidean_distance(query_node.point, furthest_.second.get().point);

                if (distance_to_neighbor < distance_to_furthest_) {
                    candidates.emplace(distance_to_neighbor, neighbor.get());
                    result_map.emplace(distance_to_neighbor, neighbor.get());
                    result_map.erase(--result_map.cend());
                }
            }
//            if (nearest.point.id == query.id) break;
        }

        // add query_node's kNN
        for (const auto& neighbor : query_node.neighbors) {
            if (checked[neighbor.get().point.id]) continue;
            checked[neighbor.get().point.id] = true;
            const auto distance_to_neighbor = euclidean_distance(query_node.point, neighbor.get().point);
            result_map.emplace(distance_to_neighbor, neighbor.get());
        }

        // result_map => result vector;
        return [&result_map, &checked_nodes]() {
            vector<reference_wrapper<const Node>> r;
            unordered_map<size_t, bool> added;

            // add result
            for (const auto& neighbor_pair : result_map) {
                added[neighbor_pair.second.get().point.id] = true;
                r.push_back(neighbor_pair.second.get());
            }

            // add checked nodes
            for (const auto& node_pair : checked_nodes) {
                if (added[node_pair.second.get().point.id]) continue;
                added[node_pair.second.get().point.id] = true;
                r.push_back(node_pair.second.get());
            }

            return r;
        }();
    }

    size_t find_navi_node_id(const Graph& knn_graph, uint n_sample, uint random_state = 42) {
        mt19937 engine(random_state);
        uniform_int_distribution<size_t> dist(0, knn_graph.size() - 1);

        // sampling
        const auto sampled = [&knn_graph, n_sample, &engine, &dist]() {
            vector<reference_wrapper<const Node>> nodes;
            unordered_map<size_t, bool> added;
            for (uint i = 0; i < n_sample; i++) {
                auto random_id = dist(engine);
                if (added[random_id]) {
                    i--;
                    continue;
                }
                added[random_id] = true;
                nodes.push_back(knn_graph[random_id]);
            }
            return nodes;
        }();

        // calc sum of distance of each points
        const auto distance_list = [&sampled, n_sample]() {
           vector<float> dl(n_sample);
#pragma omp parallel for
            for (unsigned i = 0; i < n_sample; i++) {
                const auto& node1 = sampled[i];
                float distance_sum = 0;
                for (const auto& node2 : sampled) {
                if (node1.get().point == node2.get().point) continue;
                    const auto d = euclidean_distance(node1.get().point, node2.get().point);
                    distance_sum += d;
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

    bool conflict(const Node& p, const Node& v) {
        // true if p is in v's neighbor r's neighbor (edge pr is not detour)
        for (const auto& r : v.neighbors) {
            if (r.get().point.id == p.point.id) return true;
            for (const auto& r_neighbor : r.get().neighbors) {
                if (r_neighbor.get().point.id != p.point.id) continue;

                // check if pr is not detour
                auto distance_to_p_from_v = euclidean_distance(p.point, v.point);
                auto distance_to_p_from_r = euclidean_distance(p.point, r.get().point);
                if (distance_to_p_from_v > distance_to_p_from_r) return true;
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

    NSG create_nsg(const string& data_dir, const string& knng_path, unsigned m, unsigned k,
                   unsigned n_sample, int nk) {
        // init
        auto series_for_knng = load_data(data_dir, nk);
        auto series_for_nsg = series_for_knng;

        const auto& knn_graph = load_graph(series_for_knng, knng_path, nk);

        const auto navi_node_id = find_navi_node_id(knn_graph, n_sample);
        const auto& navi_node_knng = knn_graph[navi_node_id];
        auto nsg = NSG(series_for_nsg, navi_node_id);

        cout << "complete: load data and kNNG" << endl;

        // create
        const auto checked_node_list_along_search = [&knn_graph, &navi_node_knng, k]() {
            vector<vector<reference_wrapper<const Node>>> node_list(knn_graph.size());
#pragma omp parallel for
            for (unsigned i = 0; i < knn_graph.size(); i++) {
                const auto& v = knn_graph[i];
                const auto nodes = knn_search_with_checked(v, k, navi_node_knng);
                node_list[i] = nodes;
            }
            return node_list;
        }();

        cout << "complete: calcurate kNN" << endl;

        const unsigned par = nsg.size() / 100;
        for (unsigned i = 0; i < nsg.size(); i++) {
            auto& v = nsg[i];

            if (v.point.id % par == 0) {
                float progress = v.point.id / par;
                cout << "progress: " << progress << "%" << endl;
            }

            unsigned n_add_edges = 0;
            for (const auto& p_knng : checked_node_list_along_search[i]) {
                auto& p = nsg[p_knng.get().point.id];
                if (p.point == v.point || conflict(p, v)) continue;
                v.add_neighbor(p);
                ++n_add_edges;
                if (n_add_edges >= m) break;
            }
        }

        cout << "complete: create NSG" << endl;

        while (true) {
            // check connection with dfs
            auto connected = [&nsg, navi_node_id]() {
                auto visited = unordered_map<size_t, bool>(nsg.size());
                visited[navi_node_id] = true;
                dfs(nsg.navi_node, visited);
                return visited;
            }();

            // connect disconnected node
            bool all_connected = true;
            for (const auto& node : nsg) {
                if (connected[node.point.id]) continue;
                auto& disconnected_node = nsg[node.point.id];
                const auto knn = knn_search(disconnected_node.point, k, nsg.navi_node);
                auto& nearest_to_disconnected = nsg[knn[0].get().point.id];
                disconnected_node.add_neighbor(nearest_to_disconnected);
                nearest_to_disconnected.add_neighbor(disconnected_node);
                all_connected = false;
            }
            if (all_connected) break;
        }

        cout << "complete: check connection" << endl;

        return nsg;
    }
}


#endif //NSG_NSG_HPP
