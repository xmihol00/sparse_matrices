#include <jni.h>
#include <string>
#include <arm_neon.h>
//#include "arm_neon_.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iostream>
#include <fstream>
#include "copied/models.h"
#include "copied/dense.h"

using namespace std;

short* generateRamp(short startValue, short len) {
    short* ramp = new short[len];

    for(short i = 0; i < len; i++) {
        ramp[i] = startValue + i;
    }

    return ramp;
}

double msElapsedTime(chrono::system_clock::time_point start) {
    auto end = chrono::system_clock::now();

    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

chrono::system_clock::time_point now() {
    return chrono::system_clock::now();
}

int dotProduct(short* vector1, short* vector2, short len) {
    int result = 0;

    for(short i = 0; i < len; i++) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

int dotProductNeon(short* vector1, short* vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    int32x4_t partialSumsNeon = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    int32x4_t sum4 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll with 4
    int i = 0;
    for(; i+3 < segments; i+=4) {
        // Preload may help speed up sometimes
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector1) :);
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector2) :);

        // Load vector elements to registers
        int16x8_t v11 = vld1q_s16(vector1);
        int16x4_t v11_low = vget_low_s16(v11);
        int16x4_t v11_high = vget_high_s16(v11);

        int16x8_t v12 = vld1q_s16(vector2);
        int16x4_t v12_low = vget_low_s16(v12);
        int16x4_t v12_high = vget_high_s16(v12);

        int16x8_t v21 = vld1q_s16(vector1+8);
        int16x4_t v21_low = vget_low_s16(v21);
        int16x4_t v21_high = vget_high_s16(v21);

        int16x8_t v22 = vld1q_s16(vector2+8);
        int16x4_t v22_low = vget_low_s16(v22);
        int16x4_t v22_high = vget_high_s16(v22);

        // Multiply and accumulate: partialSumsNeon += vector1Neon * vector2Neon
        sum1 = vmlal_s16(sum1, v11_low, v12_low);
        sum2 = vmlal_s16(sum2, v11_high, v12_high);
        sum3 = vmlal_s16(sum3, v21_low, v22_low);
        sum4 = vmlal_s16(sum4, v21_high, v22_high);

        vector1 += 16;
        vector2 += 16;
    }
    partialSumsNeon = sum1 + sum2 + sum3 + sum4;

    // Sum up remain parts
    int remain = len % transferSize;
    for(i=0; i<remain; i++) {

        int16x4_t vector1Neon = vld1_s16(vector1);
        int16x4_t vector2Neon = vld1_s16(vector2);
        partialSumsNeon = vmlal_s16(partialSumsNeon, vector1Neon, vector2Neon);

        vector1 += 4;
        vector2 += 4;
    }

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for(int i = 0; i < transferSize; i++) {
        result += partialSums[i];
    }

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_sparseandroidml_MainActivity_stringFromJNI(
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
    std::string path(path_cstr);
    env->ReleaseStringUTFChars(path_jstring, path_cstr);

    //filesystem::path file_path = path + "/matrix.csv";
    //std::ifstream input_file(file_path);
    //string line = "";
    //if (!input_file.is_open()) {
    //    line = "Error opening file: " + file_path.string();
    //}
    //else
    //{
    //    getline(input_file, line);
    //    input_file.close();
    //}

    Models::Mnist32x32_4L_KinMSparse<4, 16> sparseModel1(path + "/weights_", path + "/biases_");
    Matrix::Dense inputSparse1(path + "/mnist_X_test_T.csv", Matrix::COLUMN_MAJOR);
    auto start = now();
    Matrix::Dense outputSparse1 = sparseModel1.predict(inputSparse1);
    Matrix::Dense resultsSparse1 = outputSparse1.argmax(0);
    auto elapsedTime = msElapsedTime(start);
    std::string resultsString = "Elapsed time sparse 4 in 16: " + to_string((int) elapsedTime) + " ms\n";
    sparseModel1.~Mnist32x32_4L_KinMSparse();
    inputSparse1.~Dense();
    outputSparse1.~Dense();
    resultsSparse1.~Dense();

    Models::Mnist32x32_4L_4in16Sparse sparseModel2(path + "/weights_", path + "/biases_");
    Matrix::Dense inputSparse2(path + "/mnist_X_test_T.csv", Matrix::COLUMN_MAJOR);
    start = now();
    Matrix::Dense outputSparse2 = sparseModel2.predict(inputSparse2);
    Matrix::Dense resultsSparse2 = outputSparse2.argmax(0);
    elapsedTime = msElapsedTime(start);
    resultsString += "Elapsed time sparse 4 in 16: " + to_string((int) elapsedTime) + " ms\n";
    sparseModel2.~Mnist32x32_4L_4in16Sparse();
    inputSparse2.~Dense();
    outputSparse2.~Dense();
    resultsSparse2.~Dense();

    Models::Mnist32x32_4L_KinMSparse<2, 16> sparseModel3(path + "/weights_", path + "/biases_");
    Matrix::Dense inputSparse3(path + "/mnist_X_test_T.csv", Matrix::COLUMN_MAJOR);
    start = now();
    Matrix::Dense outputSparse3 = sparseModel3.predict(inputSparse3);
    Matrix::Dense resultsSparse3 = outputSparse3.argmax(0);
    elapsedTime = msElapsedTime(start);
    resultsString += "Elapsed time sparse 2 in 16: " + to_string((int) elapsedTime) + " ms\n";
    sparseModel3.~Mnist32x32_4L_KinMSparse();
    inputSparse3.~Dense();
    outputSparse3.~Dense();
    resultsSparse3.~Dense();

    Models::Mnist32x32_4L_KinMSparse<2, 16> sparseModel4(path + "/weights_", path + "/biases_");
    Matrix::Dense inputSparse4(path + "/mnist_X_test_T.csv", Matrix::COLUMN_MAJOR);
    start = now();
    Matrix::Dense outputSparse4 = sparseModel4.predictThreads(inputSparse4);
    Matrix::Dense resultsSparse4 = outputSparse4.argmax(0);
    elapsedTime = msElapsedTime(start);
    resultsString += "Elapsed time sparse 2 in 16 threads: " + to_string((int) elapsedTime) + " ms\n";
    sparseModel4.~Mnist32x32_4L_KinMSparse();
    inputSparse4.~Dense();
    outputSparse4.~Dense();
    resultsSparse4.~Dense();

    Models::Mnist32x32_4L model(path + "/weights_", path + "/biases_");
    Matrix::Dense input(path + "/mnist_X_test_T.csv", Matrix::COLUMN_MAJOR);
    start = now();
    Matrix::Dense output = model.predict(input);
    Matrix::Dense results = output.argmax(0);
    elapsedTime = msElapsedTime(start);
    resultsString += "Elapsed time dense: " + to_string((int) elapsedTime) + " ms\n";

    return env->NewStringUTF(resultsString.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_sparseandroidml_MainActivity_inference(JNIEnv* env, jobject context)
{
    return env->NewStringUTF("TEST function call");
}
