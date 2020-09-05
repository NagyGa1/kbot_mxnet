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

    input_size(1, inputs);
    output_size(1, outputs);

    // Let MXNet infer shapes other parameters such as weights
    net.InferArgsMap(ctx, &args, args);

    // Initialize all parameters with uniform distribution U(-0.01, 0.01)
    auto initializer = Uniform(0.01);
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

    // Create executor by binding parameters to the model
    exec = net.SimpleBind(ctx, args);
    arg_names = net.ListArguments();
}

MXNetwork::MXNetwork(const std::string &fname, const Context ctx) : ctx{ctx} {
    net = Symbol::Load(fname + ".json");
    NDArray::Load(fname + ".params", nullptr, &args);
    // Create executor by binding parameters to the model
    exec = net.SimpleBind(ctx, args);
}

MXNetwork::MXNetwork(const MXNetwork &other) : ctx{other.ctx} {
    this->net = other.net.Copy();
    for (const auto &arg : other.args) {
        arg.second.WaitToRead();
        this->args[arg.first] = arg.second.Copy(ctx);
    }

    // Create executor by binding parameters to the model
    exec = net.SimpleBind(ctx, args);
}

MXNetwork::~MXNetwork() {
    delete opt;
    delete exec;
}

void MXNetwork::fit(const NDArray &input, const NDArray &output) {
    auto batch_sent = input.GetShape()[0];

    batch_size_if_needed(batch_sent);

    // Set data and label
    input.WaitToRead();
    inputs().WaitToWrite();
    input.CopyTo(&inputs());

    output.WaitToRead();
    outputs().WaitToWrite();
    output.CopyTo(&outputs());

    inputs().WaitToRead();
    outputs().WaitToRead();

    fit();
}

void MXNetwork::fit() {
    // Compute gradients
    exec->Forward(true);
    exec->Backward();
    // Update parameters
    for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "input" || arg_names[i] == "output") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
    }
}

mxnet::cpp::NDArray *MXNetwork::predict(const NDArray &input) {
    input.WaitToRead();
    inputs().WaitToWrite();
    input.CopyTo(&inputs());
    inputs().WaitToRead();
    // Forward pass is enough as no gradient is needed when evaluating
    exec->Forward(false);
    exec->outputs[0].WaitToRead();
    return &exec->outputs[0];
}

void MXNetwork::save(const std::string &fname) {
    for(const auto& arg : args) arg.second.WaitToRead();

    auto args_to_save = args;
    args_to_save["input"] = NDArray(Shape(1, input_width()), ctx);
    auto output_w = output_width();
    if (output_w == 1) args_to_save["output"] = NDArray(Shape(1), ctx);
    else args_to_save["output"] = NDArray(Shape(1, output_w), ctx);
    NDArray::Save(fname + ".params", args_to_save);
    net.Save(fname + ".json");
}

void MXNetwork::batch_size(int new_batch_size) {

    input_size(new_batch_size, input_width());
    output_size(new_batch_size, output_width());

    // Create executor by binding parameters to the model
    delete exec;
    exec = net.SimpleBind(ctx, args);
}

void MXNetwork::batch_size_if_needed(int new_batch_size) {
    if (new_batch_size != batch_size()) {
#ifndef NDEBUG
        std::cout << "New batch size: " << new_batch_size << std::endl;
#endif
        batch_size(new_batch_size);
    }
}

int MXNetwork::batch_size() {
    return inputs().GetShape()[0];
}

int MXNetwork::input_width() {
    return inputs().GetShape()[1];
}

void MXNetwork::input_size(const int height, const int width) {
    inputs().WaitToWrite();
    inputs() = NDArray(Shape(height, width), ctx);
}

int MXNetwork::output_width() {
    auto output_shape = outputs().GetShape();
    if (output_shape.size() == 1) {
        return 1;
    } else {
        return output_shape.back();
    }
}

void MXNetwork::output_size(const int height, const int width) {
    outputs().WaitToWrite();
    if (width == 1) outputs() = NDArray(Shape(height), ctx);
    else outputs() = NDArray(Shape(height, width), ctx);
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
    s << "MXNetwork " << this << " [" << batch_size() << ", " << input_width() << "]";
    s << " [";
    bool first = true;
    for (const auto &arg_name : arg_names) {
        if (!first) s << ", ";
        first = false;
        s << arg_name << "/" << args[arg_name].GetShape()[0];
    }
    s << "] => [" << batch_size() << "]";
    return s.str();
}

mxnet::cpp::NDArray &MXNetwork::inputs() {
    return args["input"];
}

mxnet::cpp::NDArray &MXNetwork::outputs() {
    return args["output"];
}




