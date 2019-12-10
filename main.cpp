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
    const unsigned l = config["l"];
    const unsigned c = config["c"];
    const unsigned n_sample = config["n_sample"];
    const string df = config["distance"];

    auto nsg = NSG(df);
    nsg.build(data_path, knng_path, m, n_sample, n, l, c);
    nsg.save(save_path);
}