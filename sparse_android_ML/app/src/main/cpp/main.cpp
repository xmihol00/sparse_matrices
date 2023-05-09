#include <jni.h>
#include <string>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iostream>
#include <fstream>
#include "copied/models.h"
#include "copied/dense.h"
#include "NNAPI_models.h"

#include <android/NeuralNetworks.h>
#include <android/NeuralNetworksTypes.h>
#include <android/log.h>

using namespace std;
using namespace Matrix;
using namespace Models;

static NNAPI_Mnist32x32_4L modelNNAPI;
static Mnist32x32_4L modelDense;
static Mnist32x32_4L_Threads<8> modelDenseThreads;
static Mnist32x32_4L_KinMSparse<4, 16> model4in16;
static Mnist32x32_4L_KinMSparse<4, 16> model2in16;
static Dense X_test;
static Dense y_test;

extern "C" JNIEXPORT void JNICALL
Java_com_example_sparseandroidml_MainActivity_loadModels(
        JNIEnv* env,
        jobject context) 
{
    jclass context_class = env->GetObjectClass(context);
    jmethodID get_files_dir_method = env->GetMethodID(context_class, "getFilesDir", "()Ljava/io/File;");
    jobject file_object = env->CallObjectMethod(context, get_files_dir_method);
    jclass file_class = env->GetObjectClass(file_object);
    jmethodID get_path_method = env->GetMethodID(file_class, "getAbsolutePath", "()Ljava/lang/String;");
    jstring path_jstring = (jstring)env->CallObjectMethod(file_object, get_path_method);
    const char *path_cstr = env->GetStringUTFChars(path_jstring, nullptr);
    string path(path_cstr);
    env->ReleaseStringUTFChars(path_jstring, path_cstr);

    //model4in16.load(path + "/weights_", path + "/biases_");
    //model2in16.load(path + "/weights_", path + "/biases_");
    modelDense.load(path + "/weights_", path + "/biases_");
    modelDenseThreads.load(path + "/weights_", path + "/biases_");
    modelNNAPI.load(path);

    X_test = Dense(path + "/mnist_X_test.csv", COLUMN_MAJOR);
    y_test = Dense(path + "/mnist_y_test.csv", COLUMN_MAJOR);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseThreadsSample(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    //Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    //Dense output = modelDense.predict(input);
    //Dense result = output.argmax(0);
    uint32_t result = modelDenseThreads.predictRaw(floatArrayElements);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    //return static_cast<jint>(result(0, 0));
    return static_cast<jint>(result);
}
extern "C" JNIEXPORT jint JNICALL  Java_com_example_sparseandroidml_MainActivity_runDenseThreadsTestSet(JNIEnv* env, jobject context)
{

}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModel(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    uint8_t result = modelDense.predictOptimizedRaw(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT void JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelTestSet(JNIEnv* env, jobject context)
{
    Dense output = modelDense.predictOptimized(X_test);
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d %d %d %d", X_test.getRows(), X_test.getColumns(), y_test.getRows(), y_test.getColumns());
    output.argmax(0);
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Accuracy: %f", output.percentageDifference(y_test));
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    //Dense output = model4in16.predictOptimizedThreads<8>(input);
    Dense output = modelDense.predictOptimized(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_sparseandroidml_MainActivity_run2in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    //Dense output = model2in16.predictOptimized(input);
    Dense output = modelDense.predictOptimized(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelNNAPI(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    int result = modelNNAPI.predict(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);

    return result;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_sparseandroidml_MainActivity_testMLAPI(JNIEnv* env, jobject context)
{
    //ANEURALNETWORKS_NO_ERROR
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksModel_create(&model);

    uint32_t rows = 3;
    uint32_t cols = 3;

    float input[] = {
            1.0f, 1.0f, 1.0f,
    };

    float weights[] = {
            1.0f, 2.0f, 3.0f,
            1.0f, 2.0f, 3.0f,
            1.0f, 2.0f, 3.0f
    };
    float biases[] = {
            1.0f, 1.0f, 1.0f
    };


// Add input operands ANEURALNETWORKS_FUSED_NONE
    uint32_t inputDims[] = {1, 3};
    uint32_t weightDims[] = {3, 3};
    uint32_t biasDims[] = {3};
    uint32_t outputDims[] = {1, 3};
    ANeuralNetworksOperandType outputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = outputDims};
    ANeuralNetworksOperandType inputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = inputDims};
    ANeuralNetworksOperandType weightType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = weightDims};
    ANeuralNetworksOperandType biasType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 1, .dimensions = biasDims};
    ANeuralNetworksOperandType inputTypeScalar = {.type = ANEURALNETWORKS_INT32, .dimensionCount = 0, .dimensions = nullptr };
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &weightType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &biasType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputTypeScalar) ||
            ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &outputType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &weightType) ||
            ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &biasType) ||
            ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputTypeScalar) ||
            ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &outputType))
    {
        return env->NewStringUTF("addOperand input error");
    }


// Add the matrix multiplication operation
    uint32_t inIndexes[] = {0, 1, 2, 3};
    uint32_t outIndexes[] = {4};
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inIndexes, 1, outIndexes))
    {
        return env->NewStringUTF("addOperation error");
    }
    uint32_t inIndexes1[] = {4, 5, 6, 7};
    uint32_t outIndexes1[] = {8};
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inIndexes1, 1, outIndexes1))
    {
        return env->NewStringUTF("addOperation error");
    }

    uint32_t inputs[] = { 0, 1, 2, 3, 5, 6, 7};
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_identifyInputsAndOutputs(model, 7, inputs, 1, outIndexes1))
    {
        return env->NewStringUTF("identify error");
    }
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_finish(model))
    {
        return env->NewStringUTF("finish modelDense");
    }
    ANeuralNetworksCompilation* compilation = nullptr;
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksCompilation_create(model, &compilation))
    {
        return env->NewStringUTF("create compilation");
    }
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksCompilation_finish(compilation))
    {
        return env->NewStringUTF("finish compilation");
    }

    ANeuralNetworksExecution* execution = nullptr;
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_create(compilation, &execution))
    {
        return env->NewStringUTF("create execution");
    }

    int32_t activation = ANEURALNETWORKS_FUSED_NONE;
// Set inputs
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 0, nullptr, input, sizeof(float) * cols) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 1, nullptr, weights, sizeof(float) * rows * cols) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 2, nullptr, biases, sizeof(float) * rows) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 3, nullptr, &activation, sizeof(int32_t)) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 4, nullptr, weights, sizeof(float) * rows * cols) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 5, nullptr, biases, sizeof(float) * rows) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 6, nullptr, &activation, sizeof(int32_t)))
    {
        return env->NewStringUTF("set input");
    }

// Set output
    float* outputMatrix = new float[rows * cols]();
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setOutput(execution, 0, nullptr, outputMatrix, sizeof(float) * rows))
    {
        return env->NewStringUTF("set output");
    }

    /*ANeuralNetworksEvent* event = nullptr;
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_startCompute(execution, &event))
    {
        return env->NewStringUTF("start compute");
    }

    // Wait for the execution to complete
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksEvent_wait(event))
    {
        return env->NewStringUTF("wait");
    }*/
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_compute(execution))
    {
        return env->NewStringUTF("compute");
    }

    string resultsString = "";
    for (int i = 0; i < 1; i++)
    {
        resultsString += to_string(outputMatrix[i]) + " ";
        resultsString += to_string(input[i]) + " ";
        resultsString += to_string(weights[i]) + " ";
    }
    // Free resources
    //ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);

    delete[] outputMatrix;
    return env->NewStringUTF(resultsString.c_str());
}
