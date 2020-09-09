#include <iostream>
#include "MXNetwork.h"
#include <mxnet-cpp/MxNetCpp.h>
#include <mxnet/base.h>

using namespace mxnet::cpp;

int main() {
    std::cout << "MXNet version: " << MXNET_VERSION << std::endl;
    const std::vector<int> layers{30, 1};

    for(int i = 0; i < 100; i++) {
        if(i % 1000 == 0) std::cout << "MXNetwork() #" << i << std::endl;
        auto net = new MXNetwork(
                2, layers, ActivationActType::kRelu, "LinearRegression", "adam",
                std::map<std::string, float>{{"lr", 0.001f}},
                Context::cpu()
        );
        // Executors leak memory unfortunately.
        // These get deallocated in the MXNetwork destructor, but something still leaks.
        // All fine without this line.
        auto exec = net->exec_for_batch_size(6);
        delete net;
    }


    return 0;
}