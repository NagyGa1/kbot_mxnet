#ifndef KBOT_NATIVE_MXNETWORK_H
#define KBOT_NATIVE_MXNETWORK_H

#include <mxnet-cpp/MxNetCpp.h>
#include <mxnet/base.h>

class MXNetwork {
public:
    /**
     * Constructor for training.
     *
     * The output size is the last hidden_layer size.
     *
     * @param inputs
     * @param hidden_layers
     * @param activation_type
     * @param output_type
     * @param optimizer name of the optimizer for training, e.g. "adam".
     * @param ctx
     */
    explicit MXNetwork(int inputs,
                       const std::vector<int> &hidden_layers,
                       mxnet::cpp::ActivationActType activation_type,
                       const std::string &output_type,
                       const std::string &optimizer,
                       const std::map<std::string, float> &optimizer_params,
                       mxnet::cpp::Context ctx);

    /**
     * Construct by loading previously trained network.
     * Networks are saved with batch_size = 1.
     *
     * @param fname
     * @param ctx
     */
    explicit MXNetwork(const std::string &fname, mxnet::cpp::Context ctx);

    MXNetwork(const MXNetwork &);

    MXNetwork &operator=(const MXNetwork &) = delete;

    MXNetwork(const MXNetwork &&) = delete;

    MXNetwork &&operator=(const MXNetwork &&) = delete;

    ~MXNetwork();

    /**
     * Updates batch size if differs from input.
     *
     * @param input
     * @param output
     */
    void fit(const mxnet::cpp::NDArray &input, const mxnet::cpp::NDArray &output);

    /**
     * Fit assuming input and output was updated.
     */
    void fit(mxnet::cpp::Executor* exec);

    /**
     * Assumes proper batch size set previously, matching input!
     * Decided to this for performance reasons.
     *
     * @param input
     * @return
     */
    mxnet::cpp::NDArray *predict(const mxnet::cpp::NDArray &input);

    /**
     * Saves all NDArrays and the Symbol structure to .params and .json files.
     * Input and output is saved with batch_size set to 1, to save space - so after loading that will be
     * the default.
     * Optimizers are not saved.
     *
     * @param fname
     */
    void save(const std::string &fname);

    mxnet::cpp::Executor* exec_for_batch_size(int batch_size);

    [[maybe_unused]] void print_args();

    std::string to_string();

    mxnet::cpp::Context ctx;

private:
    mxnet::cpp::Symbol net;
    /**
     * Arrays for batch size = 1.
     */
    std::map<std::string, mxnet::cpp::NDArray> args;
    mxnet::cpp::Optimizer *opt{};
    std::map<int, mxnet::cpp::Executor*> executors_for_batch_sizes;
};


#endif //KBOT_NATIVE_MXNETWORK_H
