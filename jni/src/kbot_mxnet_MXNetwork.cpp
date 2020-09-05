#include <jni.h>
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include "MXNetwork.h"

#ifndef _Included_kbot_mxnet_MXNetwork
#define _Included_kbot_mxnet_MXNetwork
#ifdef __cplusplus
extern "C" {
#endif

jfieldID MXNetwork_net;

using namespace mxnet::cpp;

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_nativeInit
        (JNIEnv *env, jclass clazz) {
#ifndef NDEBUG
    std::cout << "nativeInit " << std::endl;
#endif

    MXNetwork_net = env->GetFieldID(clazz, "net", "J");
    if (env->ExceptionCheck()) return;

}

JNIEXPORT jstring JNICALL Java_kbot_mxnet_MXNetwork_kbot_1native_1version
        (JNIEnv *env, jclass) {
    std::stringstream s;
    s << KBOT_NATIVE_MAJOR << "." << KBOT_NATIVE_MINOR << "." << KBOT_NATIVE_PATCH;
    return env->NewStringUTF(s.str().c_str());
}

JNIEXPORT jstring JNICALL Java_kbot_mxnet_MXNetwork_mxnet_1version
        (JNIEnv *env, jclass) {
    std::stringstream s;
    s << MXNET_MAJOR << "." << MXNET_MINOR << "." << MXNET_PATCH;
    return env->NewStringUTF(s.str().c_str());
}

void throwRTE(JNIEnv *env, const std::string &message) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), message.c_str());
}

JNIEXPORT jlong JNICALL Java_kbot_mxnet_MXNetwork_constructNet
        (JNIEnv *env, jobject, jint inputs,
         jintArray hidden_layers, jstring activation_type, jstring output_type, jstring optimizer_type,
         jobjectArray optimizer_param_names, jfloatArray optimizer_param_values,
         jint context_device_type, jint context_device_id) {

    auto hls = env->GetIntArrayElements(hidden_layers, nullptr);
    if (env->ExceptionCheck()) return 0;
    std::vector<int> hlsv(hls, &hls[env->GetArrayLength(hidden_layers)]);
    env->ReleaseIntArrayElements(hidden_layers, hls, JNI_ABORT);
    if (env->ExceptionCheck()) return 0;

    auto output_type2 = env->GetStringUTFChars(output_type, nullptr);
    if (env->ExceptionCheck()) return 0;
    auto activation_type2 = env->GetStringUTFChars(activation_type, nullptr);
    if (env->ExceptionCheck()) return 0;
    auto optimizer_type2 = env->GetStringUTFChars(optimizer_type, nullptr);
    if (env->ExceptionCheck()) return 0;

    std::string act_type2 = activation_type2;
    ActivationActType act_type;
    if (act_type2 == "Relu") {
        act_type = ActivationActType::kRelu;
    } else if (act_type2 == "Sigmoid") {
        act_type = ActivationActType::kSigmoid;
    } else if (act_type2 == "Softrelu") {
        act_type = ActivationActType::kSoftrelu;
    } else if (act_type2 == "Softsign") {
        act_type = ActivationActType::kSoftsign;
    } else if (act_type2 == "Tanh") {
        act_type = ActivationActType::kTanh;
    } else {
        throwRTE(env, "Unknown activation type: " + act_type2);
        return 0;
    }

    auto optimizer_param_len = env->GetArrayLength(optimizer_param_names);
    std::map<std::string, float> optimizer_params;
    for (int i = 0; i < optimizer_param_len; i++) {
        auto param_name = (jstring) env->GetObjectArrayElement(optimizer_param_names, i);
        if (env->ExceptionCheck()) return 0;
        auto param_name_s = env->GetStringUTFChars(param_name, nullptr);
        float f;
        env->GetFloatArrayRegion(optimizer_param_values, i, 1, &f);
        if (env->ExceptionCheck()) return 0;
        optimizer_params[std::string(param_name_s)] = f;
        env->ReleaseStringUTFChars(param_name, param_name_s);
    }

    auto net = new MXNetwork(
            inputs, hlsv, act_type, std::string(output_type2),
            std::string(optimizer_type2),
            optimizer_params,
            Context(static_cast<const DeviceType>(context_device_type), context_device_id)
    );

    env->ReleaseStringUTFChars(output_type, output_type2);
    if (env->ExceptionCheck()) return 0;
    env->ReleaseStringUTFChars(activation_type, activation_type2);
    if (env->ExceptionCheck()) return 0;
    env->ReleaseStringUTFChars(optimizer_type, optimizer_type2);
    if (env->ExceptionCheck()) return 0;

#ifndef NDEBUG
    std::cout << "constructNet " << net->to_string() << std::endl;
#endif
    return (jlong) net;
}

JNIEXPORT jlong JNICALL Java_kbot_mxnet_MXNetwork_constructNetLoad
        (JNIEnv *env, jobject, jstring file_name, jint context_device_type, jint context_device_id) {
    auto file_name2 = env->GetStringUTFChars(file_name, nullptr);
    auto fname = std::string(file_name2);
    env->ReleaseStringUTFChars(file_name, file_name2);

    auto net = new MXNetwork(fname, Context(static_cast<const DeviceType>(context_device_type), context_device_id));

#ifndef NDEBUG
    std::cout << "constructNetLoad " << net->to_string() << " << " << fname << std::endl;
#endif
    return (jlong) net;
}

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_disposeNet
        (JNIEnv *env, jobject obj) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);
#ifndef NDEBUG
    std::cout << "destructNet " << net->to_string() << std::endl;
#endif
    delete net;
    env->SetLongField(obj, MXNetwork_net, 0);
}

JNIEXPORT jlong JNICALL Java_kbot_mxnet_MXNetwork_cloneNet
        (JNIEnv *env, jobject obj) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);
    auto clone = new MXNetwork(*net);
#ifndef NDEBUG
    std::cout << "cloneNet " << clone << " <- " << net << std::endl;
#endif
    return (jlong) clone;
}

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_save
        (JNIEnv *env, jobject obj, jstring file_name) {
    auto file_name2 = env->GetStringUTFChars(file_name, nullptr);
    auto fname = std::string(file_name2);
    env->ReleaseStringUTFChars(file_name, file_name2);

    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);
    net->save(fname);
}

/**
 *
 * @param env
 * @param inputs
 * @param nda
 * @param input_from_idx
 * @param rows this many rows to copy
 * @return success
 */
void copyToNDArray(JNIEnv *env, jobjectArray &inputs, NDArray &nda, const int input_from_idx, const int rows) {
    auto cols = env->GetArrayLength((jfloatArray) env->GetObjectArrayElement(inputs, 0));

    auto shape = nda.GetShape();
    if (shape.size() != 2 || shape[0] != rows || shape[1] != cols) {
        std::stringstream ss;
        ss << "Shape mismatch: size " << shape.size() << " (" << shape[0] << "," << shape[1]
           << ")  != (" << rows << "," << cols << ")";
        throwRTE(env, ss.str());
        return;
    }

//    NDArray tmp{Shape(rows, cols), mxnet::cpp::Context::cpu(), false};

    nda.WaitToWrite();

    for (int i = 0; i < rows; i++) {
        // this kind of copy only works on CPU.
        const float *target = nda.GetData() + nda.Offset(i, 0);
//        const float *target = tmp.GetData() + tmp.Offset(i, 0);
        env->GetFloatArrayRegion(
                (jfloatArray) env->GetObjectArrayElement(inputs, input_from_idx + i),
                0, cols,
                (jfloat *) target
        );
    }

//    tmp.CopyTo(&nda);
}

JNIEXPORT jfloatArray JNICALL Java_kbot_mxnet_MXNetwork_predict1D_1m
        (JNIEnv *env, jobject obj, jobjectArray inputs) {

    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);

    auto inputsLen = env->GetArrayLength(inputs);
    net->batch_size_if_needed(inputsLen);
    copyToNDArray(env, inputs, net->inputs(), 0, env->GetArrayLength(inputs));
    if (env->ExceptionCheck()) return nullptr;

    net->inputs().WaitToRead();

    net->exec->Forward(false);

    net->exec->outputs[0].WaitToRead();

    auto input_rows = env->GetArrayLength(inputs);
    jfloatArray ret = env->NewFloatArray(input_rows);
    // this kind of copy only works on CPU.
    env->SetFloatArrayRegion(ret, 0, input_rows, net->exec->outputs[0].GetData());
    return ret;
}

JNIEXPORT jfloat JNICALL Java_kbot_mxnet_MXNetwork_predict1D
        (JNIEnv *env, jobject obj, jfloatArray input) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);
    auto cols = env->GetArrayLength(input);

    net->batch_size_if_needed(1);

    NDArray *net_inputs = &net->inputs();

    auto shape = net_inputs->GetShape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] != cols) {
        std::stringstream ss;
        ss << "Shape mismatch: size " << shape.size() << " (" << shape[0] << "," << shape[1]
           << ")  != (" << 1 << "," << cols << ")";
        throwRTE(env, ss.str());
        return std::numeric_limits<float>::quiet_NaN();
    }

    net_inputs->WaitToWrite();
    env->GetFloatArrayRegion(input, 0, cols, (jfloat *) net_inputs->GetData());
    net_inputs->WaitToRead();

    net->exec->Forward(false);
    NDArray *outs = &net->exec->outputs[0];
    auto out_cols = outs->GetShape()[1];

    if (out_cols != 1) {
        std::stringstream ss;
        ss << "Output shape mismatch: size " << shape.size() << " (" << shape[0] << "," << shape[1]
           << ")  != (" << 1 << "," << 1 << ")";
        throwRTE(env, ss.str());
        return std::numeric_limits<float>::quiet_NaN();
    }

    outs->WaitToRead();

    return outs->At(0, 0);
}

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_fit1D
        (JNIEnv *env, jobject obj, jobjectArray inputs, jfloatArray outputs) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);

    auto inputsLen = env->GetArrayLength(inputs);
    net->batch_size_if_needed(inputsLen);
    copyToNDArray(env, inputs, net->inputs(), 0, env->GetArrayLength(inputs));
    if (env->ExceptionCheck()) return;

    net->outputs().WaitToWrite();

    auto outputsLen = env->GetArrayLength(outputs);
//    NDArray tmp{Shape(outputsLen), mxnet::cpp::Context::cpu(), false};
//    env->GetFloatArrayRegion(outputs, 0, outputsLen, (float *) tmp.GetData());
//    tmp.CopyTo(&net->outputs());
    env->GetFloatArrayRegion(outputs, 0, outputsLen, (float *) net->outputs().GetData()); // CPU only
    if (env->ExceptionCheck()) return;

    net->inputs().WaitToRead();
    net->outputs().WaitToRead();

    net->fit();

    NDArray::WaitAll();
}

JNIEXPORT jobjectArray JNICALL Java_kbot_mxnet_MXNetwork_predict2D_1m
        (JNIEnv *env, jobject obj, jobjectArray inputs) {

    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);

    auto input_rows = env->GetArrayLength(inputs);
    net->batch_size_if_needed(input_rows);
    copyToNDArray(env, inputs, net->inputs(), 0, env->GetArrayLength(inputs));
    if (env->ExceptionCheck()) return nullptr;
    net->inputs().WaitToRead();

    net->exec->Forward(false);

    net->exec->outputs[0].WaitToRead();

    auto output_shape = net->exec->outputs[0].GetShape();
    if (output_shape.size() != 2) {
        throwRTE(env, "output_shape.size() != 2");
        return nullptr;
    }

    auto outs = &net->exec->outputs[0];
    jfloatArray fas[output_shape[0]];
    for (int row = 0; row < output_shape[0]; row++) {

        jfloatArray fa = env->NewFloatArray(output_shape[1]);
        if (env->ExceptionCheck()) return nullptr;
        fas[row] = fa;

        // this kind of copy only works on CPU.
        env->SetFloatArrayRegion(fa, 0, output_shape[1], &outs->GetData()[outs->Offset(row, 0)]);
        if (env->ExceptionCheck()) return nullptr;
    }

    jobjectArray ret = env->NewObjectArray(input_rows, env->GetObjectClass(fas[0]), nullptr);
    if (env->ExceptionCheck()) return nullptr;

    for (int row = 0; row < output_shape[0]; row++) {
        env->SetObjectArrayElement(ret, row, fas[row]);
        if (env->ExceptionCheck()) return nullptr;
    }

    return ret;
}

JNIEXPORT jfloatArray JNICALL Java_kbot_mxnet_MXNetwork_predict2D
        (JNIEnv *env, jobject obj, jfloatArray input) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);
    auto cols = env->GetArrayLength(input);

    net->batch_size_if_needed(1);

    NDArray *net_inputs = &net->inputs();

    auto shape = net_inputs->GetShape();
    if (shape.size() != 2 || shape[0] != 1 || shape[1] != cols) {
        std::stringstream ss;
        ss << "Shape mismatch: size " << shape.size() << " (" << shape[0] << "," << shape[1]
           << ")  != (" << 1 << "," << cols << ")";
        throwRTE(env, ss.str());
        return nullptr;
    }

    net_inputs->WaitToWrite();
    env->GetFloatArrayRegion(input, 0, cols, (jfloat *) net_inputs->GetData());
    net_inputs->WaitToRead();

    net->exec->Forward(false);
    NDArray *outs = &net->exec->outputs[0];
    auto out_cols = outs->GetShape()[1];

    jfloatArray ret = env->NewFloatArray(out_cols);
    if (env->ExceptionCheck()) return nullptr;

    outs->WaitToRead();
    env->SetFloatArrayRegion(ret, 0, out_cols, outs->GetData());

    return ret;
}

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_fit2D
        (JNIEnv *env, jobject obj, jobjectArray inputs, jobjectArray outputs) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);

    auto inputs_len = env->GetArrayLength(inputs);
    net->batch_size_if_needed(inputs_len);

    copyToNDArray(env, inputs, net->inputs(), 0, env->GetArrayLength(inputs));
    if (env->ExceptionCheck()) return;

    copyToNDArray(env, outputs, net->outputs(), 0, env->GetArrayLength(outputs));
    if (env->ExceptionCheck()) return;

    net->inputs().WaitToRead();
    net->outputs().WaitToRead();

    net->fit();
}

JNIEXPORT void JNICALL Java_kbot_mxnet_MXNetwork_fit2D_1mb
        (JNIEnv *env, jobject obj, jobjectArray inputs, jobjectArray outputs, jint batch_size) {
    auto net = (MXNetwork *) env->GetLongField(obj, MXNetwork_net);

    auto inputs_len = env->GetArrayLength(inputs);

    for (int start = 0; start < inputs_len; start += batch_size) {
        auto elements_to_send = std::min(inputs_len - start, batch_size);

        net->batch_size_if_needed(elements_to_send);

        copyToNDArray(env, inputs, net->inputs(), start, elements_to_send);
        if (env->ExceptionCheck()) return;

        copyToNDArray(env, outputs, net->outputs(), start, elements_to_send);
        if (env->ExceptionCheck()) return;

        net->inputs().WaitToRead();
        net->outputs().WaitToRead();

        net->fit();
    }

}

#ifdef __cplusplus
}
#endif
#endif
