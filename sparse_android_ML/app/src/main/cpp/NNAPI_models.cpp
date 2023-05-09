#include "NNAPI_models.h"

using namespace std;
using namespace Matrix;
using namespace Models;

NNAPI_Mnist32x32_4L::NNAPI_Mnist32x32_4L() : _output{10, 1, COLUMN_MAJOR}
{

}

NNAPI_Mnist32x32_4L::NNAPI_Mnist32x32_4L(string path) : NNAPI_Mnist32x32_4L()
{
    load(path);
}

NNAPI_Mnist32x32_4L::~NNAPI_Mnist32x32_4L()
{
    ANeuralNetworksCompilation_free(_compilation);
    ANeuralNetworksModel_free(_model);

    ANeuralNetworksMemory_free(_B0SharedMemory);
    ANeuralNetworksMemory_free(_B1SharedMemory);
    ANeuralNetworksMemory_free(_B2SharedMemory);
    ANeuralNetworksMemory_free(_B3SharedMemory);
    ANeuralNetworksMemory_free(_B4SharedMemory);

    ANeuralNetworksMemory_free(_W0SharedMemory);
    ANeuralNetworksMemory_free(_W1SharedMemory);
    ANeuralNetworksMemory_free(_W2SharedMemory);
    ANeuralNetworksMemory_free(_W3SharedMemory);
    ANeuralNetworksMemory_free(_W4SharedMemory);
}

void NNAPI_Mnist32x32_4L::loadToSharedMemory(std::string name, ANeuralNetworksMemory *&memory, Matrix::Dense &matrix)
{
    int fd = ASharedMemory_create(name.c_str(), matrix.getSize());
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%s fd: %d", name.c_str(), fd);
    int status = ANeuralNetworksMemory_createFromFd(matrix.getSize(), PROT_READ | PROT_WRITE, fd, 0, &memory);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to create shared memory for %s", name.c_str());
    }

    float *mappedMemory = reinterpret_cast<float *>(mmap(nullptr, matrix.getSize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    memcpy(mappedMemory, matrix.getData(), matrix.getSize());
    munmap(mappedMemory, matrix.getSize());
}

void NNAPI_Mnist32x32_4L::load(string path)
{
    _W0 = Dense(path + "/weights_l0.csv", ROW_MAJOR);
    _W1 = Dense(path + "/weights_l1.csv", ROW_MAJOR);
    _W2 = Dense(path + "/weights_l2.csv", ROW_MAJOR);
    _W3 = Dense(path + "/weights_l3.csv", ROW_MAJOR);
    _W4 = Dense(path + "/weights_l4.csv", ROW_MAJOR);

    _B0 = Dense(path + "/biases_l0.csv", COLUMN_MAJOR);
    _B1 = Dense(path + "/biases_l1.csv", COLUMN_MAJOR);
    _B2 = Dense(path + "/biases_l2.csv", COLUMN_MAJOR);
    _B3 = Dense(path + "/biases_l3.csv", COLUMN_MAJOR);
    _B4 = Dense(path + "/biases_l4.csv", COLUMN_MAJOR);

    size_t hiddenWeightsSize = 1024 * 1024 * sizeof(float);
    size_t outputWeightsSize = 10 * 1024 * sizeof(float);
    size_t hiddenBiasesSize = 1024 * sizeof(float);
    size_t outputBiasesSize = 10 * sizeof(float);

    int status = ANeuralNetworksModel_create(&_model);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to create model");
    }

    uint32_t hiddenWeightsDimensions[2] = { 1024U, 1024U };
    ANeuralNetworksOperandType hiddenWeightsOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 2,
            .dimensions = hiddenWeightsDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    loadToSharedMemory("W0", _W0SharedMemory, _W0);
    loadToSharedMemory("W1", _W1SharedMemory, _W1);
    loadToSharedMemory("W2", _W2SharedMemory, _W2);
    loadToSharedMemory("W3", _W3SharedMemory, _W3);
    loadToSharedMemory("W4", _W4SharedMemory, _W4);

    loadToSharedMemory("B0", _B0SharedMemory, _B0);
    loadToSharedMemory("B1", _B1SharedMemory, _B1);
    loadToSharedMemory("B2", _B2SharedMemory, _B2);
    loadToSharedMemory("B3", _B3SharedMemory, _B3);
    loadToSharedMemory("B4", _B4SharedMemory, _B4);

    uint32_t hiddenBiasesDimensions[1] = { 1024U };
    ANeuralNetworksOperandType hiddenBiasesOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 1,
            .dimensions = hiddenBiasesDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t outputWeightsDimensions[2] = { 10U, 1024U };
    ANeuralNetworksOperandType outputWeightsOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 2,
            .dimensions = outputWeightsDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t outputBiasesDimensions[1] = { 10U };
    ANeuralNetworksOperandType outputBiasesOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 1,
            .dimensions = outputBiasesDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    ANeuralNetworksOperandType activationFunctionOperand{
            .type = ANEURALNETWORKS_INT32,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t inputDimensions[2] = { 1U, 1024U };
    ANeuralNetworksOperandType inputOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 2,
            .dimensions = inputDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t outputDimensions[2] = { 1U, 10U };
    ANeuralNetworksOperandType outputOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 2,
            .dimensions = outputDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t intermediateOutputDimensions[2] = { 1U,1024U };
    ANeuralNetworksOperandType intermediateOutputOperand{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 2,
            .dimensions = intermediateOutputDimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    status = ANeuralNetworksModel_addOperand(_model, &inputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add input operand");
    }

    status = ANeuralNetworksModel_addOperand(_model, &hiddenWeightsOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden weights l0 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &hiddenBiasesOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden biases l0 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &activationFunctionOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add ReLU l0 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &intermediateOutputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add intermediate output l0 operand");
    }
    
    status = ANeuralNetworksModel_addOperand(_model, &hiddenWeightsOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden weights l1 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &hiddenBiasesOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden biases l1 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &activationFunctionOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add ReLU l1 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &intermediateOutputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add intermediate output l1 operand");
    }
        
    status = ANeuralNetworksModel_addOperand(_model, &hiddenWeightsOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden weights l2 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &hiddenBiasesOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden biases l2 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &activationFunctionOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add ReLU l2 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &intermediateOutputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add intermediate output l2 operand");
    }
    
    status = ANeuralNetworksModel_addOperand(_model, &hiddenWeightsOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden weights l3 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &hiddenBiasesOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add hidden biases l3 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &activationFunctionOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add ReLU l3 operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &intermediateOutputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add intermediate output l3 operand");
    }
        
    status = ANeuralNetworksModel_addOperand(_model, &outputWeightsOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add output weights operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &outputBiasesOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add output biases operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &activationFunctionOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add NONE activation function operand");
    }
    status = ANeuralNetworksModel_addOperand(_model, &outputOperand);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add output operand");
    }

    // indices:
    // 0 - input
    // 1 - weights l0
    // 2 - biases l0
    // 3 - activation function ReLU
    // 4 - intermediate output l0
    // 5 - weights l1
    // 6 - biases l1
    // 7 - activation function ReLU
    // 8 - intermediate output l1
    // 9 - weights l2
    // 10 - biases l2
    // 11 - activation function ReLU
    // 12 - intermediate output l2
    // 13 - weights l3
    // 14 - biases l3
    // 15 - activation function ReLU
    // 16 - intermediate output l3
    // 17 - weights output
    // 18 - biases output
    // 19 - activation function NONE
    // 20 - output

    //status = ANeuralNetworksModel_setOperandValue(_model, 1, _W0.getData(), _W0.getSize());

    /*float *w01Ptr = reinterpret_cast<float *>(mmap(nullptr, hiddenWeightsSize, PROT_READ, MAP_SHARED, _W0fd, 0));
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "W0SharedMemory[0]: %f", w01Ptr[0]);
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "W0SharedMemory[1]: %f", w01Ptr[1]);
    __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "W0SharedMemory[2]: %f", w01Ptr[2]);
    munmap(w01Ptr, hiddenWeightsSize);*/
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 1, _W0SharedMemory, 0, hiddenWeightsSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set W0", status);
    }
    //status = ANeuralNetworksModel_setOperandValue(_model, 2, _B0.getData(), _B0.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 2, _B0SharedMemory, 0, hiddenBiasesSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set B0", status);
    }
    status = ANeuralNetworksModel_setOperandValue(_model, 3, &_activationReLU, sizeof(int32_t));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set ReLU activation", status);
    }

    //status = ANeuralNetworksModel_setOperandValue(_model, 5, _W1.getData(), _W1.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 5, _W1SharedMemory, 0, hiddenWeightsSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set W1", status);
    }
    //status = ANeuralNetworksModel_setOperandValue(_model, 6, _B1.getData(), _B1.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 6, _B1SharedMemory, 0, hiddenBiasesSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set B1", status);
    }
    status = ANeuralNetworksModel_setOperandValue(_model, 7, &_activationReLU, sizeof(int32_t));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set ReLU activation", status);
    }

    //status = ANeuralNetworksModel_setOperandValue(_model, 9, _W2.getData(), _W2.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 9, _W2SharedMemory, 0, hiddenWeightsSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set W2", status);
    }
    //status = ANeuralNetworksModel_setOperandValue(_model, 10, _B2.getData(), _B2.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 10, _B2SharedMemory, 0, hiddenBiasesSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set B2", status);
    }
    status = ANeuralNetworksModel_setOperandValue(_model, 11, &_activationReLU, sizeof(int32_t));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set ReLU activation", status);
    }

    //status = ANeuralNetworksModel_setOperandValue(_model, 13, _W3.getData(), _W3.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 13, _W3SharedMemory, 0, hiddenWeightsSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set W3", status);
    }
    //status = ANeuralNetworksModel_setOperandValue(_model, 14, _B3.getData(), _B3.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 14, _B3SharedMemory, 0, hiddenBiasesSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set B3", status);
    }
    status = ANeuralNetworksModel_setOperandValue(_model, 15, &_activationReLU, sizeof(int32_t));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set ReLU activation", status);
    }

    //status = ANeuralNetworksModel_setOperandValue(_model, 17, _W4.getData(), _W4.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 17, _W4SharedMemory, 0, outputWeightsSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set W4", status);
    }
    //status = ANeuralNetworksModel_setOperandValue(_model, 18, _B4.getData(), _B4.getSize());
    status = ANeuralNetworksModel_setOperandValueFromMemory(_model, 18, _B4SharedMemory, 0, outputBiasesSize);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set B4", status);
    }
    status = ANeuralNetworksModel_setOperandValue(_model, 19, &_activationNONE, sizeof(int32_t));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set NONE activation", status);
    }

    // indices:
    // 0 - input
    // 1 - weights l0
    // 2 - biases l0
    // 3 - activation function ReLU
    // 4 - intermediate output l0
    // 5 - weights l1
    // 6 - biases l1
    // 7 - activation function ReLU
    // 8 - intermediate output l1
    // 9 - weights l2
    // 10 - biases l2
    // 11 - activation function ReLU
    // 12 - intermediate output l2
    // 13 - weights l3
    // 14 - biases l3
    // 15 - activation function ReLU
    // 16 - intermediate output l3
    // 17 - weights output
    // 18 - biases output
    // 19 - activation function NONE
    // 20 - output

    uint32_t layer0InIndexes[] = { 0, 1, 2, 3 };
    uint32_t layer0OutIndexes[] = { 4 };
    status = ANeuralNetworksModel_addOperation(_model, ANEURALNETWORKS_FULLY_CONNECTED, 4, layer0InIndexes, 1, layer0OutIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add l0 FULLY_CONNECTED operation");
    }

    uint32_t layer1InIndexes[] = { 4, 5, 6, 7 };
    uint32_t layer1OutIndexes[] = { 8 };
    status = ANeuralNetworksModel_addOperation(_model, ANEURALNETWORKS_FULLY_CONNECTED, 4, layer1InIndexes, 1, layer1OutIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add l1 FULLY_CONNECTED operation");
    }
    
    uint32_t layer2InIndexes[] = { 8, 9, 10, 11 };
    uint32_t layer2OutIndexes[] = { 12 };
    status = ANeuralNetworksModel_addOperation(_model, ANEURALNETWORKS_FULLY_CONNECTED, 4, layer2InIndexes, 1, layer2OutIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add l2 FULLY_CONNECTED operation");
    }
    
    uint32_t layer3InIndexes[] = { 12, 13, 14, 15 };
    uint32_t layer3OutIndexes[] = { 16 };
    status = ANeuralNetworksModel_addOperation(_model, ANEURALNETWORKS_FULLY_CONNECTED, 4, layer3InIndexes, 1, layer3OutIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add l3 FULLY_CONNECTED operation");
    }

    uint32_t layer4InIndexes[] = { 16, 17, 18, 19 };
    uint32_t layer4OutIndexes[] = { 20 };
    status = ANeuralNetworksModel_addOperation(_model, ANEURALNETWORKS_FULLY_CONNECTED, 4, layer4InIndexes, 1, layer4OutIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add l4 FULLY_CONNECTED operation");
    }

    uint32_t inIndexes[] = { 0 };
    uint32_t outIndexes[] = { 20 };
    status = ANeuralNetworksModel_identifyInputsAndOutputs(_model, sizeof(inIndexes) / sizeof(outIndexes), inIndexes, 1, outIndexes);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to add identify inputs and outputs");
    }

    status = ANeuralNetworksModel_finish(_model);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to finish model");
    }

    status = ANeuralNetworksCompilation_create(_model, &_compilation);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Compilation creation failed");
    }

    status = ANeuralNetworksCompilation_setPreference(_compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Compilation set preference failed");
    }

    status = ANeuralNetworksCompilation_finish(_compilation);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Compilation finish failed");
    }
}

uint8_t NNAPI_Mnist32x32_4L::predict(float *input)
{

    int status = ANeuralNetworksExecution_create(_compilation, &_execution);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Execution creation failed");
    }
    status = ANeuralNetworksExecution_setInput(_execution, 0, nullptr, input, INPUT_SIZE);
    
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "%d: Failed to set input 0", status);
    }
    status = ANeuralNetworksExecution_setOutput(_execution, 0, nullptr, _output.getData(), _output.getSize());
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to set output");
    }

    status = ANeuralNetworksExecution_compute(_execution);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        __android_log_print(ANDROID_LOG_ERROR, "NNAPI_Mnist32x32_4L", "Failed to compute");
    }

    ANeuralNetworksExecution_free(_execution);
    return _output.argmax();
}
