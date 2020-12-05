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

    float outputs[]{0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F};

    const std::vector<int> layers{30, 1};

    auto net = MXNetwork(
            2, layers, ActivationActType::kRelu, "LinearRegression", "adam",
            std::map<std::string, float>{{"lr", 0.001f}},
            Context::cpu()
    );

    MXNetwork *netCopy;
    float netCopyChkSum = 0.0f;

    NDArray input{reinterpret_cast<const mx_float *>(&features), Shape{6, 2}, Context::cpu()};
    NDArray output{reinterpret_cast<const mx_float *>(&outputs), Shape{6}, Context::cpu()};

    auto t_start = std::chrono::high_resolution_clock::now();

    const auto NUM_EPOCHS = 50000;
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {

        net.fit(input, output);

        NDArray *pred = net.predict(input);

        float preds[6];
        pred->SyncCopyToCPU(preds);

//        std::cout << "pred: #" << epoch;
//        for (float p : preds) {
//            std::cout << " " << p;
//        }
//        std::cout << std::endl;

        if (epoch == 1) {
            netCopy = new MXNetwork(net);
            for (auto p : preds) netCopyChkSum += p;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    std::cout << NUM_EPOCHS << " epochs in " << (duration / 1000000.0) << " seconds" << std::endl;
    std::cout << (1.0 / (duration / 1000000.0)) * NUM_EPOCHS << " epochs per second" << std::endl;

    return 0;
}