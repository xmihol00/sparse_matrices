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

static Mnist32x32_4L_KinMSparse<4, 16> model4in16;
static Mnist32x32_4L_KinMSparse<2, 16> model2in16;

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
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_sparseandroidml_MainActivity_run4in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    Dense output = model4in16.predict(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_sparseandroidml_MainActivity_run2in16model(JNIEnv* env, jobject context, jfloatArray sample)
{
    float *floatArrayElements = env->GetFloatArrayElements(sample, nullptr);

    Dense input(1024, 1, Matrix::COLUMN_MAJOR, reinterpret_cast<byte *>(floatArrayElements));
    Dense output = model2in16.predict(input);
    Dense result = output.argmax(0);

    env->ReleaseFloatArrayElements(sample, floatArrayElements, 0);
    return static_cast<jint>(result(0, 0));
}
