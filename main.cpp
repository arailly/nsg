#include <iostream>
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;

int main() {
    auto config = read_config();
    string data_path = config["data_path"];
    string knng_path = config["knng_path"];
    string save_path = config["save_path"];
    unsigned n = config["n"];
    unsigned m = config["m"];
    unsigned k = config["k"];
    unsigned n_sample = config["n_sample"];

    auto nsg = create_nsg(data_path, knng_path, m, k, n_sample, n);
    write_graph(save_path, nsg);
}