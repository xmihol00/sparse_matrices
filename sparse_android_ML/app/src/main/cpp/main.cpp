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

// declaration of the models
static NNAPI_Mnist32x32_4L modelNNAPI;
static Mnist32x32_4L modelDense;
static Mnist32x32_4L_Threads<8> modelDenseThreads;
static Mnist32x32_4L_KinMSparse<4, 16, 8> model4in16;
static Mnist32x32_4L_KinMSparse<2, 16, 8> model2in16;
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

    bool metadataFirst = false;
    // load the models
    model4in16.load(path + "/weights_", path + "/biases_", metadataFirst);
    model2in16.load(path + "/weights_", path + "/biases_", metadataFirst);
    modelDense.load(path + "/weights_", path + "/biases_");
    modelDenseThreads.load(path + "/weights_", path + "/biases_");
    modelNNAPI.load(path);

    // load MNIST test set
    X_test = Dense(path + "/mnist_X_test_T.csv", COLUMN_MAJOR);
    y_test = Dense(path + "/mnist_y_test.csv", COLUMN_MAJOR);
}


extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModel(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint8_t result = modelDense.predictOptimizedRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = modelDense.predictOptimizedMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseThreadsSample(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint32_t result = modelDenseThreads.predictRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL  Java_com_example_sparseandroidml_MainActivity_runDenseThreadsTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = modelDenseThreads.predictMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint32_t result = model4in16.predictRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16modelTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = model4in16.predictMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run2in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint32_t result = model2in16.predictRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_run2in16modelTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = model2in16.predictMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16modelThreads(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint32_t result = model4in16.predictThreadsRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_run4in16modelThreadsTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = model4in16.predictThreadsMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_run2in16modelThreads(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    uint32_t result = model2in16.predictThreadsRawSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result);
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_run2in16modelThreadsTestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = model2in16.predictThreadsMatrix(X_test);
    return output.percentageDifference(y_test);
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelNNAPI(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);
    // predict without copying memory
    int result = modelNNAPI.predictSample(floatArrayElements);
    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);

    return result;
}

extern "C" JNIEXPORT jfloat JNICALL Java_com_example_sparseandroidml_MainActivity_runDenseModelNNAPITestSet(JNIEnv* env, jobject context)
{
    // predict the already loaded test set
    Dense output = modelNNAPI.predictTestSet(X_test.getData(), 10'000);
    return output.percentageDifference(y_test);
}
