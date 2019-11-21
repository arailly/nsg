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
    uint n = config["n"];
    uint m = config["m"];
    uint k = config["k"];

    auto nsg = create_nsg(data_path, knng_path, m, k, n);
    write_graph(save_path, nsg);
}