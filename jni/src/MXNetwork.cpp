#include "MXNetwork.h"

using namespace mxnet::cpp;

MXNetwork::MXNetwork(
        const int inputs,
        const std::vector<int> &hidden_layers,
        const ActivationActType activation_type,
        const std::string &output_type,
        const std::string &optimizer,
        const std::map<std::string, float> &optimizer_params,
        const Context ctx
) : ctx{ctx} {

    auto outputs = hidden_layers.back();

    auto x = Symbol::Variable("input");
    auto label = Symbol::Variable("output");

    std::vector<Symbol> weights(hidden_layers.size());
    std::vector<Symbol> biases(hidden_layers.size());
    std::vector<Symbol> outs(hidden_layers.size());

    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        weights[i] = Symbol::Variable("w" + std::to_string(i));
        biases[i] = Symbol::Variable("b" + std::to_string(i));
        Symbol fc = FullyConnected(
                i == 0 ? x : outs[i - 1],  // data
                weights[i],
                biases[i],
                hidden_layers[i]);
        outs[i] = i == hidden_layers.size() - 1 ? fc : Activation(fc, activation_type);
    }

    if (output_type == "Softmax") {
        net = SoftmaxOutput(outs.back(), label);
    } else if (output_type == "LinearRegression") {
        net = LinearRegressionOutput(outs.back(), label);
    } else if (output_type == "MAERegression") {
        net = MAERegressionOutput(outs.back(), label);
    } else {
        throw std::runtime_error("Unknown output type: " + output_type);
    }

    // initialize input and output arrays with batch size = 1
    args["input"] = NDArray(Shape(1, inputs), ctx);
    if (outputs == 1) args["output"] = NDArray(Shape(1), ctx);
    else args["output"] = NDArray(Shape(1, outputs), ctx);

    // Let MXNet infer shapes other parameters such as weights
    net.InferArgsMap(ctx, &args, args);

    // Initialize all parameters
    auto initializer = Xavier();
    for (auto &arg : args) {
        // arg.first is parameter name, and arg.second is the value
        initializer(arg.first, &arg.second);
    }

    // Create sgd optimizer
//    opt = OptimizerRegistry::Find("sgd");
//    opt->SetParam("rescale_grad", 1.0 / batch_size)
//            ->SetParam("lr", learning_rate)
//            ->SetParam("wd", weight_decay);

    opt = OptimizerRegistry::Find(optimizer);
    for (const auto &opt_param : optimizer_params) {
#ifndef NDEBUG
        std::cout << "Optimizer parameter set: " << opt_param.first << " = " << opt_param.second << std::endl;
#endif
        opt->SetParam(opt_param.first, opt_param.second);
    }
}

MXNetwork::MXNetwork(const std::string &fname, const Context ctx) : ctx{ctx} {
    net = Symbol::Load(fname + ".json");
    NDArray::Load(fname + ".params", nullptr, &args);
}

MXNetwork::MXNetwork(const MXNetwork &other) : ctx{other.ctx} {
    this->net = other.net.Copy();
    for (const auto &arg : other.args) {
        arg.second.WaitToRead();
        this->args[arg.first] = arg.second.Copy(ctx);
    }
}

MXNetwork::~MXNetwork() {
    delete opt;
    for (auto exec : executors_for_batch_sizes) {
        delete exec.second;
    }
}

void MXNetwork::fit(const NDArray &input, const NDArray &output) {
    auto batch_size = input.GetShape().at(0);
    auto exec = exec_for_batch_size(batch_size);

    auto exec_in = &exec->arg_arrays.front();
    auto exec_out = &exec->arg_arrays.back();

    // Set data and label
    input.WaitToRead();
    exec_in->WaitToWrite();
    input.CopyTo(exec_in);

    output.WaitToRead();
    exec_out->WaitToWrite();
    output.CopyTo(exec_out);

    exec_in->WaitToRead();
    exec_out->WaitToRead();

    fit(exec);
}

void MXNetwork::fit(mxnet::cpp::Executor *exec) {
    // Compute gradients
    exec->Forward(true);
    exec->Backward();

    // Update parameters. Skipping first and lasts as these should be input the and output.
    for (size_t i = 1; i < exec->arg_arrays.size() - 1; ++i) {
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
    }
}

mxnet::cpp::NDArray *MXNetwork::predict(const NDArray &input) {
    auto batch_size = input.GetShape()[0];
    auto exec = exec_for_batch_size(batch_size);

    auto exec_in = &exec->arg_arrays.front();

    input.WaitToRead();
    exec_in->WaitToWrite();
    input.CopyTo(exec_in);
    exec_in->WaitToRead();

    // Forward pass is enough as no gradient is needed when evaluating
    exec->Forward(false);
    exec->outputs[0].WaitToRead();
    return &exec->outputs[0];
}

void MXNetwork::save(const std::string &fname) {
    for (const auto &arg : args) arg.second.WaitToRead();

    // always save the batch size = 1 arrays, which is args.
    NDArray::Save(fname + ".params", args);
    net.Save(fname + ".json");
}

Executor *MXNetwork::exec_for_batch_size(int batch_size) {
    auto exec = executors_for_batch_sizes.find(batch_size);

    if (exec == executors_for_batch_sizes.end()) {
#ifndef NDEBUG
        std::cout << "New executor for batch size: " << batch_size << std::endl;
#endif

        auto input_shape = args.at("input").GetShape();
        auto output_shape = args.at("output").GetShape();

        CHECK(input_shape.size() == 2);
        CHECK(output_shape.size() == 1 || output_shape.size() == 2);

        auto args_for_new_batch_size = args;
        args_for_new_batch_size.at("input") = NDArray(Shape(batch_size, input_shape.at(1)), ctx);

        if (output_shape.size() == 1) {
            args_for_new_batch_size.at("output") = NDArray(Shape(batch_size), ctx, false);
        } else {
            args_for_new_batch_size.at("output") = NDArray(Shape(batch_size, output_shape.at(1)), ctx, false);
        }
        auto new_exec = net.SimpleBind(ctx, args_for_new_batch_size);

        // check if first is input last is output, assumed later in code.

        CHECK(new_exec->arg_arrays.front().GetData() == args_for_new_batch_size["input"].GetData());
        CHECK(new_exec->arg_arrays.back().GetData() == args_for_new_batch_size["output"].GetData());

        executors_for_batch_sizes[batch_size] = new_exec;
        return new_exec;
    } else {
        return exec->second;
    }
}

[[maybe_unused]] void MXNetwork::print_args() {
    for (const auto &pair : args) {
        std::cout << pair.first << " [";
        bool first = true;
        for (auto i : pair.second.GetShape()) {
            if (!first) std::cout << ", ";
            first = false;
            std::cout << i;
        }
        std::cout << "]" << std::endl;
    }
}

std::string MXNetwork::to_string() {
    std::stringstream s;
    s << "MXNetwork " << this << " [batch, " << args.at("input").GetShape().at(1) << "]";
    s << " [";
    bool first = true;
    for (const auto& arg : args) {
        if (!first) s << ", ";
        first = false;
        s << arg.first << "/" << arg.second.GetShape().at(0);
    }
    s << "] => [batch]";
    return s.str();
}





