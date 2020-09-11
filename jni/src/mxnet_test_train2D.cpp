#include <iostream>
#include "MXNetwork.h"
#include <mxnet-cpp/MxNetCpp.h>
#include <mxnet/base.h>

using namespace mxnet::cpp;

int main() {
    std::cout << "MXNet version: " << MXNET_VERSION << std::endl;

    float features[6][2]{
            {1.0F, 1.1F},
            {2.0F, 2.2F},
            {6.0F, 3.3F},
            {6.0F, 4.4F},
            {2.5F, 2.5F},
            {5.5F, 4.6F}
    };

    float outputs[6][2]{
            {0.0F, 0.5F},
            {0.0F, 0.4F},
            {1.0F, 1.0F},
            {1.0F, 1.0F},
            {0.0F, 0.6F},
            {1.0F, 1.0F}
    };

    const std::vector<int> layers{30, 2};

    auto net = MXNetwork(
            2, layers, ActivationActType::kRelu, "LinearRegression", "adam",
            std::map<std::string, float>{{"lr", 0.001f}},
            Context::cpu()
    );

    MXNetwork *netCopy;
    float netCopyChkSum = 0.0f;

    NDArray input{reinterpret_cast<const mx_float *>(&features), Shape{6, 2}, Context::cpu()};
    NDArray output{reinterpret_cast<const mx_float *>(&outputs), Shape{6, 2}, Context::cpu()};

    for (int epoch = 1; epoch <= 100; epoch++) {

        net.fit(input, output);

        NDArray *pred = net.predict(input);

        std::cout << "pred: #" << epoch;

        float preds[6][2];
        pred->SyncCopyToCPU((float *) preds);
        for (auto p : preds) {
            std::cout << " (" << p[0] << " x " << p[1] << ")";
        }
//        for(int row = 0; row < 6; row++) {
//            std::cout << " (" << pred->At(row, 0) << " x " << pred->At(row, 1) << ")";
//        }

        std::cout << std::endl;

        if (epoch == 1) {
            netCopy = new MXNetwork(net);
            for (auto p : preds) netCopyChkSum += p[0] + p[1];
        }
    }

    std::cout << "1st epoch clone predict output: ";

    NDArray *pred = netCopy->predict(input);

    float preds[6][2];
    pred->SyncCopyToCPU((float *) preds);

    float netCopyChkSum2 = 0.0f;
    for (auto p : preds) {
        netCopyChkSum2 += p[0] + p[1];
        std::cout << " (" << p[0] << " x " << p[1] << ")";
    }
    std::cout << std::endl;

    if (netCopyChkSum == netCopyChkSum2) {
        std::cout << "Checksum matches, copy constructor verified." << std::endl;
    } else {
        std::cout << "Checksum DOES NOT MATCH, copy constructor FAIL." << std::endl;
    }

    delete netCopy;

    return 0;
}