#include <jni.h>
#include <string>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iostream>
#include <fstream>
#include "copied/models.h"
#include "copied/dense.h"

using namespace std;
using namespace Matrix;
using namespace Models;

static Mnist32x32_4L model;
static Mnist32x32_4L_Threads<4> modelThreads;
static Mnist32x32_4L_KinMSparse<4, 16> model4in16;
static Mnist32x32_4L_KinMSparse<4, 16> model2in16;

#include <android/NeuralNetworks.h>
#include <android/NeuralNetworksTypes.h>

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

    model4in16.load(path + "/weights_", path + "/biases_");
    model2in16.load(path + "/weights_", path + "/biases_");
    model.load(path + "/weights_", path + "/biases_");
    modelThreads.load(path + "/weights_", path + "/biases_");
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModel(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    //Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    //Dense output = model.predict(input);
    //Dense result = output.argmax(0);
    uint32_t result = modelThreads.predictRaw(floatArrayElements);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    //return static_cast<jint>(result(0, 0));
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelOptimized(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    Dense output = model.predictOptimized(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    Dense output = model4in16.predictOptimizedThreads<8>(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_sparseandroidml_MainActivity_run2in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    Dense output = model2in16.predictOptimized(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_sparseandroidml_MainActivity_testMLAPI(JNIEnv* env, jobject context)
{
    //ANEURALNETWORKS_NO_ERROR
    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksModel_create(&model);
// Assume inputMatrixA and inputMatrixB are float arrays representing the input matrices
    uint32_t rows = 3;
    uint32_t cols = 3;

    float inputMatrixA[] = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
    };

    float inputMatrixB[] = {
            9.0f, 8.0f, 7.0f,
            6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f
    };


// Add input operands ANEURALNETWORKS_FUSED_NONE
    uint32_t inputDims[] = {rows, cols};
    ANeuralNetworksOperandType inputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = inputDims};
    ANeuralNetworksOperandType inputTypeScalar = {.type = ANEURALNETWORKS_INT32, .dimensionCount = 0, .dimensions = nullptr };
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputType) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &inputTypeScalar))
    {
        return env->NewStringUTF("addOperand input error");
    }

// Add output operand
    uint32_t outputDims[] = {rows, cols};
    ANeuralNetworksOperandType outputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = outputDims};
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperand(model, &outputType))
    {
        return env->NewStringUTF("addOperand output error");
    }

// Add the matrix multiplication operation
    uint32_t inIndexes[] = {0, 1, 2};
    uint32_t outIndexes[] = {3};
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, inIndexes, 1, outIndexes))
    {
        return env->NewStringUTF("addOperation error");
    }

    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_identifyInputsAndOutputs(model, 3, inIndexes, 1, outIndexes))
    {
        return env->NewStringUTF("identify error");
    }
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksModel_finish(model))
    {
        return env->NewStringUTF("finish model");
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
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 0, nullptr, inputMatrixA, sizeof(float) * rows * cols) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 1, nullptr, inputMatrixB, sizeof(float) * rows * cols) ||
        ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setInput(execution, 2, nullptr, &activation, sizeof(int32_t)))
    {
        return env->NewStringUTF("set input");
    }

// Set output
    float* outputMatrix = new float[rows * cols]();
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_setOutput(execution, 0, nullptr, outputMatrix, sizeof(float) * rows * cols))
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
        resultsString += to_string(inputMatrixA[i]) + " ";
        resultsString += to_string(inputMatrixB[i]) + " ";
    }
    // Free resources
    //ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);

    delete[] outputMatrix;
    return env->NewStringUTF(resultsString.c_str());
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_example_sparseandroidml_MainActivity_testABS(JNIEnv *env, jobject context) {
    // Create a model
    ANeuralNetworksModel *model = nullptr;
    ANeuralNetworksModel_create(&model);

    // Define input tensor
    uint32_t rows = 20;
    uint32_t cols = 20;
    uint32_t inputDims[] = {rows * cols};
    ANeuralNetworksOperandType inputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                            .dimensionCount = 1,
                                            .dimensions = inputDims};
    ANeuralNetworksModel_addOperand(model, &inputType);

    // Define output tensor
    ANeuralNetworksOperandType outputType = {.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                             .dimensionCount = 1,
                                             .dimensions = inputDims};
    ANeuralNetworksModel_addOperand(model, &outputType);

    // Add the ABS operation
    uint32_t inIndexes[] = {0};
    uint32_t outIndexes[] = {1};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ABS, 1, inIndexes, 1, 0);

    // Identify inputs and outputs
    ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, inIndexes, 1, 0);

    // Finish the model
    ANeuralNetworksModel_finish(model);

    // Compile the model
    ANeuralNetworksCompilation *compilation = nullptr;
    ANeuralNetworksCompilation_create(model, &compilation);
    ANeuralNetworksCompilation_finish(compilation);

    // Create execution
    ANeuralNetworksExecution *execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);

    // Set input
    float *inputMatrixA = new float[rows * cols];
    for (int i = 1; i <= rows * cols; i++)
    {
        inputMatrixA[i-1] = -i;
    }
    ANeuralNetworksExecution_setInput(execution, 0, nullptr, inputMatrixA, sizeof(float) * rows * cols);

    // Set output
    float *outputMatrix = new float[rows * cols]();
    for (int i = 0; i < rows * cols; i++)
    {
        outputMatrix[i] = 1;
    }
    ANeuralNetworksExecution_setOutput(execution, 0, nullptr, outputMatrix, sizeof(float) * rows * cols);

    // Start execution
    //ANeuralNetworksEvent *event = nullptr;
    //ANeuralNetworksExecution_startCompute(execution, &event);
    //ANeuralNetworksBurst *burst;
    //ANeuralNetworksBurst_create(compilation, &burst);
    //ANeuralNetworksExecution_burstCompute(execution, burst);
    if (ANEURALNETWORKS_NO_ERROR != ANeuralNetworksExecution_compute(execution))
    {
        return env->NewStringUTF("compute failed");
    }

    // Wait for execution to complete
    //ANeuralNetworksEvent_wait(event);

    // Convert output matrix to string
    std::string result = "";
    for (int i = 0; i < 2; i++) {
        result += std::to_string(outputMatrix[i]) + " ";
    }

    // Free resources
    //ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    delete[] outputMatrix;
    delete[] inputMatrixA;

    return env->NewStringUTF(result.c_str());
}
