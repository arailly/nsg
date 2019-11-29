#include <iostream>
#include <arailib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace arailib;

int main() {
    const auto config = read_config();
    const string data_path = config["data_path"];
    const string knng_path = config["knng_path"];
    const string save_path = config["save_path"];
    const unsigned n = config["n"];
    const unsigned m = config["m"];
    const unsigned k = config["k"];
    const unsigned n_sample = config["n_sample"];

    const auto nsg = create_nsg(data_path, knng_path, m, k, n_sample, n);
    write_graph(save_path, nsg);
}