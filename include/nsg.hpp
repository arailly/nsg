//
// Created by Yusuke Arai on 2019/11/08.
//

#ifndef NSG_NSG_HPP
#define NSG_NSG_HPP

#include <map>
#include <unordered_map>
#include <arailib.hpp>

using namespace std;
using namespace arailib;

struct SearchResult {
    vector<Neighbor> result;
    time_t time = 0;
    unsigned long n_node_access = 0;
    unsigned long n_dist_calc = 0;
    unsigned long n_hop = 0;
    double dist_from_navi = 0;
    double recall = 0;

    Neighbors all_candidates;
};

struct SearchResults {
    vector<SearchResult> results;
    void push_back(const SearchResult& result) { results.push_back(result); }

    void save(const string& log_path, const string& result_path) {
        ofstream log_ofs(log_path);
        string line = "time,n_node_access,n_dist_calc,n_hop,dist_from_navi,recall";
        log_ofs << line << endl;

        ofstream result_ofs(result_path);
        line = "query_id,data_id,dist";
        result_ofs << line << endl;

        int query_id = 0;
        for (const auto& result : results) {
            line = to_string(result.time) + "," +
                   to_string(result.n_node_access) + "," +
                   to_string(result.n_dist_calc) + "," +
                   to_string(result.n_hop) + "," +
                   to_string(result.dist_from_navi) + "," +
                   to_string(result.recall);
            log_ofs << line << endl;

            for (const auto& neighbor : result.result) {
                line = to_string(query_id) + "," +
                       to_string(neighbor.id) + "," +
                       to_string(neighbor.dist);
                result_ofs << line << endl;
            }

            query_id++;
        }
    }
};

struct Node {
    int id;
    Data<> data;
    Neighbors neighbors;
    unordered_map<size_t, bool> added;

    void init() { added[id] = true; }
    Node() : data(Data<>(0, {0})) { init(); }
    Node(const Data<>& data) : id(data.id), data(data) { init(); }

    void add_neighbor(double dist, int neighbor_id) {
        if (added.find(neighbor_id) != added.end()) return;
        added[neighbor_id] = true;
        neighbors.emplace_back(dist, neighbor_id);
    }

    void clear_neighbor() {
        added.clear();
        neighbors.clear();
        added[id] = true;
    }
};

struct NSG {
    vector<Node> nodes;
    int navi_node_id;

    int m;
    int l_construct;
    int c_construct;

    string dist_kind;
    DistanceFunction<> calc_dist;
    mt19937 engine;

    NSG(int m, int l_construct = 40, int c_construct = 500, string dist_kind = "euclidean") :
            l_construct(l_construct), c_construct(c_construct),
            dist_kind(dist_kind), calc_dist(select_distance(dist_kind)),
            engine(mt19937(42)) {}

    void init_nodes(const Dataset<>& series) {
        for (const auto& point : series) nodes.emplace_back(point);
    }

    void load(const Dataset<>& series, const string& graph_path, int n) {
        init_nodes(series);
        // csv
        if (is_csv(graph_path)) {
            ifstream ifs(graph_path);
            if (!ifs) throw runtime_error("Can't open file!");

            string line; getline(ifs, line);
            navi_node_id = stoi(line);

            while (getline(ifs, line)) {
                const auto&& ids = split<size_t>(line);
                nodes[ids[0]].add_neighbor(0, ids[1]);
            }
            return;
        }

        // dir
        const string navi_node_path = graph_path + "/navi-node.csv";
        ifstream navi_node_ifs(navi_node_path);
        if (!navi_node_ifs) throw runtime_error("Can't open file!");

        string navi_node_line; getline(navi_node_ifs, navi_node_line);
        navi_node_id = stoi(navi_node_line);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const string path = graph_path + "/" + to_string(i) + ".csv";
            ifstream ifs(path);
            string line;
            while (getline(ifs, line)) {
                const auto ids = split<size_t>(line);
                nodes[ids[0]].add_neighbor(0, ids[1]);
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

    void load_aknng(const Dataset<>& series, const string& graph_path, int n) {
        init_nodes(series);

        // csv file
        if (is_csv(graph_path)) {
            ifstream ifs(graph_path);
            if (!ifs) {
                const string message = "Can't open file!: " + graph_path;
                throw runtime_error(message);
            }

            string line;
            while (getline(ifs, line)) {
                const auto row = split<double>(line);
                auto& node = nodes[row[0]];
                node.add_neighbor(row[2], row[1]);
            }
            return;
        }

        // dir
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const string path = graph_path + "/" + to_string(i) + ".csv";
            ifstream ifs(path);

            if (!ifs) {
                const string message = "Can't open file!: " + path;
                throw runtime_error(message);
            }

            string line;
            while (getline(ifs, line)) {
                const auto row = split<double>(line);
                auto& node = nodes[row[0]];
                node.add_neighbor(row[2], row[1]);
            }
        }
    }

    void load_aknng(const string& data_path, const string& graph_path, int n) {
        auto series = load_data(data_path, n);
        load(series, graph_path, n);
    }

    void save(const string& save_dir) {
        const string navi_node_path = save_dir + "/navi-node.csv";
        ofstream navi_node_ofs(navi_node_path);
        // write navigating node
        navi_node_ofs << navi_node_id << endl;

        // write connection
        vector<string> lines(ceil(nodes.size() / 1000.0));
        for (const auto& node : nodes) {
            const unsigned line_i = node.id / 1000;
            for (const auto& neighbor : node.neighbors) {
                lines[line_i] += to_string(node.id) + ',' +
                                 to_string(neighbor.id) + '\n';
            }
        }

        for (int i = 0; i < lines.size(); i++) {
            const string path = save_dir + "/" + to_string(i) + ".csv";
            ofstream ofs(path);
            ofs << lines[i];
        }
    }

    auto knn_search(const Data<>& query, int k, int l) {
        auto result = SearchResult();
        const auto start_time = get_now();

        vector<bool> checked(nodes.size()), added(nodes.size());
        added[navi_node_id];

        vector<Neighbor> candidates;
        const auto& navi_node = nodes[navi_node_id];
        const auto dist_from_navi = calc_dist(query, navi_node.data);
        candidates.emplace_back(dist_from_navi, navi_node_id);

        result.dist_from_navi = dist_from_navi;

        while (true) {
            // find the first unchecked node
            int first_unchecked_index = 0;
            for (const auto candidate : candidates) {
                if (!checked[candidate.id]) break;
                ++first_unchecked_index;
            }

            // checked all candidates
            if (first_unchecked_index >= l) break;

            ++result.n_hop;

            const auto first_unchecked_node_id = candidates[first_unchecked_index].id;
            checked[first_unchecked_node_id] = true;
            const auto& first_unchecked_node = nodes[first_unchecked_node_id];

            for (const auto& neighbor : first_unchecked_node.neighbors) {
                result.n_node_access++;

                if (added[neighbor.id]) continue;
                added[neighbor.id] = true;

                result.n_dist_calc++;

                const auto& neighbor_node = nodes[neighbor.id];
                const auto dist = calc_dist(query, neighbor_node.data);
                candidates.emplace_back(dist, neighbor.id);
                result.all_candidates.emplace_back(dist, neighbor.id);
            }

            // sort and resize candidates l
            sort(candidates.begin(), candidates.end(),
                 [](const auto& n1, const auto& n2) { return n1.dist < n2.dist; });
            if (candidates.size() > l) candidates.resize(l);
        }

        for (const auto& c : candidates) {
            result.result.emplace_back(c);
            if (result.result.size() >= k) break;
        }

        const auto end_time = get_now();
        result.time = get_duration(start_time, end_time);

        return result;
    }

    auto calc_neighbor_candidates(const Node& query_node) {
        Neighbors result;

        vector<bool> added(nodes.size());
        added[query_node.id] = true;

        // add query_node's neighbors
        for (const auto& neighbor : query_node.neighbors) {
            if (added[neighbor.id]) continue;
            added[neighbor.id] = true;
            result.emplace_back(neighbor);
        }

        // add checked nodes through search into vector
        auto search_result = knn_search(query_node.data, l_construct, l_construct);
        auto& all_candidates = search_result.all_candidates;
        sort_neighbors(all_candidates);
        if (all_candidates.size() > c_construct) all_candidates.resize(c_construct);

        for (const auto& candidate : all_candidates) {
            if (added[candidate.id]) continue;
            added[candidate.id] = true;
            result.emplace_back(candidate);
        }

        return result;
    }

    auto find_navi_node_id(uint n_sample) {
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
                    if (node1.get().data == node2.get().data) continue;
                    const auto dist = calc_dist(node1.get().data, node2.get().data);
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
            if (r.id == p.id) return true;
            const auto& r_node = nodes[r.id];
            for (const auto& r_neighbor : r_node.neighbors) {
                if (r_neighbor.id != p.id) continue;

                // check if pr is not detour
                auto dist_from_v_to_p = calc_dist(p.data, v.data);
                auto dist_from_v_to_r = calc_dist(r_node.data, v.data);
                auto dist_from_r_to_p = calc_dist(p.data, r_node.data);
                if (dist_from_v_to_r < dist_from_v_to_p &&
                        dist_from_v_to_p > dist_from_r_to_p) return true;
            }
        }
        return false;
    }

    void dfs(int node_id, vector<bool>& visited) {
        const auto& node = nodes[node_id];
        for (const auto& neighbor : node.neighbors) {
            if (visited[neighbor.id]) continue;
            visited[neighbor.id] = true;
            dfs(neighbor.id, visited);
        }
    }

    void build(const string& data_path, const string& aknng_path, int n) {
        load_aknng(data_path, aknng_path, n);

        navi_node_id = find_navi_node_id(1000);
        cout << "complete: load data and AKNNG" << endl;

        vector<Neighbors> neighbor_candidate_list(nodes.size());
#pragma omp parallel for
        for (int node_id = 0; node_id < nodes.size(); ++node_id) {
            const auto& node = nodes[node_id];
            neighbor_candidate_list[node_id] = calc_neighbor_candidates(node);
        }

        cout << "complete: calculate neighbor candidates" << endl;

        for (unsigned node_id = 0; node_id < nodes.size(); node_id++) {
            auto& node = nodes[node_id];

            // calculate new neighbors
            Neighbors new_neighbors;
            for (const auto& neighbor : neighbor_candidate_list[node_id]) {
                const auto& p = nodes[neighbor.id];
                if (p.id == node.id || conflict(node, p)) continue;
                new_neighbors.emplace_back(neighbor);
                if (new_neighbors.size() >= m) break;
            }

            // assign new neighbors
            node.neighbors = new_neighbors;
        }

        cout << "complete: create NSG" << endl;

        while (true) {
            // check connected flag
            vector<bool> connected(nodes.size());
            dfs(navi_node_id, connected);

            // connect disjoint node
            bool all_connected = true;
            for (auto& node : nodes) {
                if (connected[node.id]) continue;

                // if disconnected
                all_connected = false;
                const auto nn = knn_search(node.data, 1, l_construct).result[0];
                nodes[nn.id].add_neighbor(nn.dist, node.id);
            }
            if (all_connected) break;
        }

        cout << "complete: check connection" << endl;
    }
};


#endif //NSG_NSG_HPP
