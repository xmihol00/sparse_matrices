#ifndef MODELS_H
#define MODELS_H

#include <thread>
#include <barrier>
#include <semaphore>

#include "dense.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"
#include "activations.h"

namespace Models
{
    class Mnist32x32_4L
    {
        private:
            Matrix::Dense _W0;
            Matrix::Dense _W1;
            Matrix::Dense _W2;
            Matrix::Dense _W3;
            Matrix::Dense _W4;

            Matrix::Dense _B0;
            Matrix::Dense _B1;
            Matrix::Dense _B2;
            Matrix::Dense _B3;
            Matrix::Dense _B4;

            Matrix::Dense _input;
            Matrix::Dense _outputSample;
            Matrix::Dense _tmp1Sample;
            Matrix::Dense _tmp2Sample;
            Matrix::Dense _outputMatrix;
            Matrix::Dense _tmp1Matrix;
            Matrix::Dense _tmp2Matrix;

        public:
            Mnist32x32_4L() = default;
            Mnist32x32_4L(std::string weightsFileTemplate, std::string biasesFileTemplate);
            ~Mnist32x32_4L();

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate);

            void predict(Matrix::Dense &input, Matrix::Dense &output);
            Matrix::Dense predict(Matrix::Dense &input);

            Matrix::Dense predictOptimizedMatrix(Matrix::Dense &input);
            uint8_t predictOptimizedRawSample(float *input);
    };

    template <uint8_t numberOfThreads>
    class Mnist32x32_4L_Threads
    {
        private:
            Matrix::Dense _W0;
            Matrix::Dense _W1;
            Matrix::Dense _W2;
            Matrix::Dense _W3;
            Matrix::Dense _W4;

            Matrix::Dense _B0;
            Matrix::Dense _B1;
            Matrix::Dense _B2;
            Matrix::Dense _B3;
            Matrix::Dense _B4;

            Matrix::Dense _input;
            Matrix::Dense _outputSample;
            Matrix::Dense _tmp1Sample;
            Matrix::Dense _tmp2Sample;
            Matrix::Dense _outputMatrix;
            Matrix::Dense _tmp1Matrix;
            Matrix::Dense _tmp2Matrix;

            std::counting_semaphore<numberOfThreads - 1> _semaphore;
            std::barrier<> _barrier;
            std::thread _threads[numberOfThreads - 1];
            bool _run;
            bool _predictRow;

            void predictThread(uint8_t threadId)
            {
                using namespace Matrix;

                _semaphore.acquire();        
                while (_run)
                {
                    if (_predictRow)
                    {
                        _W0.dotAddActivateRowThread(_input, _B0, _tmp1Sample, ReLU, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W1.dotAddActivateRowThread(_tmp1Sample, _B1, _tmp2Sample, ReLU, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W2.dotAddActivateRowThread(_tmp2Sample, _B2, _tmp1Sample, ReLU, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W3.dotAddActivateRowThread(_tmp1Sample, _B3, _tmp2Sample, ReLU, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W4.dotAddActivateRowThread(_tmp2Sample, _B4, _outputSample, identity, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();
                    }
                    else
                    {
                        _W0.dotAddActivateColumnThread(_input, _B0, _tmp1Matrix, ReLU, numberOfThreads, threadId);
                        _W1.dotAddActivateColumnThread(_tmp1Matrix, _B1, _tmp2Matrix, ReLU, numberOfThreads, threadId);
                        _W2.dotAddActivateColumnThread(_tmp2Matrix, _B2, _tmp1Matrix, ReLU, numberOfThreads, threadId);
                        _W3.dotAddActivateColumnThread(_tmp1Matrix, _B3, _tmp2Matrix, ReLU, numberOfThreads, threadId);
                        _W4.dotAddActivateColumnThread(_tmp2Matrix, _B4, _outputMatrix, identity, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();
                    }

                    _semaphore.acquire();
                }
            }

        public:
            Mnist32x32_4L_Threads() : _semaphore(0), _barrier(numberOfThreads), _run(true), _predictRow(true)
            {
                using namespace std;

                for (uint8_t i = 1; i < numberOfThreads; i++)
                {
                    _threads[i - 1] = thread(&Mnist32x32_4L_Threads::predictThread, this, i);
                }
            }

            ~Mnist32x32_4L_Threads()
            {
                _input.clear();
                _run = false;
                _semaphore.release(numberOfThreads - 1);
                for (uint8_t i = 0; i < numberOfThreads - 1; i++)
                {
                    _threads[i].join();
                }
            }

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate)
            {
                using namespace Matrix;

                _W0 = Dense{weightsFileTemplate + "l0.csv", ROW_MAJOR, 0, numberOfThreads};
                _W1 = Dense{weightsFileTemplate + "l1.csv", ROW_MAJOR, 0, numberOfThreads};
                _W2 = Dense{weightsFileTemplate + "l2.csv", ROW_MAJOR, 0, numberOfThreads};
                _W3 = Dense{weightsFileTemplate + "l3.csv", ROW_MAJOR, 0, numberOfThreads};
                _W4 = Dense{weightsFileTemplate + "l4.csv", ROW_MAJOR, 0, numberOfThreads};

                _B0 = Dense{biasesFileTemplate + "l0.csv", COLUMN_MAJOR, 0, numberOfThreads};
                _B1 = Dense{biasesFileTemplate + "l1.csv", COLUMN_MAJOR, 0, numberOfThreads};
                _B2 = Dense{biasesFileTemplate + "l2.csv", COLUMN_MAJOR, 0, numberOfThreads};
                _B3 = Dense{biasesFileTemplate + "l3.csv", COLUMN_MAJOR, 0, numberOfThreads};
                _B4 = Dense{biasesFileTemplate + "l4.csv", COLUMN_MAJOR, 0, numberOfThreads};

                _outputSample = Dense(_B4.getRows(), 1, COLUMN_MAJOR, numberOfThreads);
                _tmp1Sample = Dense(_B0.getRows(), 1, COLUMN_MAJOR, numberOfThreads);
                _tmp2Sample = Dense(_B0.getRows(), 1, COLUMN_MAJOR, numberOfThreads);

                _outputMatrix = Dense(_B4.getRows(), 10'000, COLUMN_MAJOR, numberOfThreads);
                _tmp1Matrix = Dense(_B0.getRows(), 10'000, COLUMN_MAJOR, numberOfThreads);
                _tmp2Matrix = Dense(_B0.getRows(), 10'000, COLUMN_MAJOR, numberOfThreads);
            }

            uint32_t predictRawSample(float *input)
            {
                using namespace Matrix;

                _predictRow = true;
                _input.setFloatMatrix(input, _W0.getColumns(), 1, COLUMN_MAJOR);
                _semaphore.release(numberOfThreads - 1);

                _W0.dotAddActivateRowThread(_input, _B0, _tmp1Sample, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W1.dotAddActivateRowThread(_tmp1Sample, _B1, _tmp2Sample, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W2.dotAddActivateRowThread(_tmp2Sample, _B2, _tmp1Sample, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W3.dotAddActivateRowThread(_tmp1Sample, _B3, _tmp2Sample, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W4.dotAddActivateRowThread(_tmp2Sample, _B4, _outputSample, identity, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                return _outputSample.argmax();
            }

            Matrix::Dense predictMatrix(Matrix::Dense input)
            {
                using namespace Matrix;

                _predictRow = false;
                _input.setFloatMatrix(input.getData(), input.getRows(), input.getColumns(), COLUMN_MAJOR);
                _semaphore.release(numberOfThreads - 1);

                _W0.dotAddActivateColumnThread(_input, _B0, _tmp1Matrix, ReLU, numberOfThreads, 0);
                _W1.dotAddActivateColumnThread(_tmp1Matrix, _B1, _tmp2Matrix, ReLU, numberOfThreads, 0);
                _W2.dotAddActivateColumnThread(_tmp2Matrix, _B2, _tmp1Matrix, ReLU, numberOfThreads, 0);
                _W3.dotAddActivateColumnThread(_tmp1Matrix, _B3, _tmp2Matrix, ReLU, numberOfThreads, 0);
                _W4.dotAddActivateColumnThread(_tmp2Matrix, _B4, _outputMatrix, identity, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                return _outputMatrix.argmax(0);
            }
    };

    class Mnist32x32_4L_4in16Sparse
    {
        private:
            Matrix::Block4in16Sparse _W0;
            Matrix::Block4in16Sparse _W1;
            Matrix::Block4in16Sparse _W2;
            Matrix::Block4in16Sparse _W3;
            Matrix::Dense _W4;

            Matrix::Dense _B0;
            Matrix::Dense _B1;
            Matrix::Dense _B2;
            Matrix::Dense _B3;
            Matrix::Dense _B4;

        public:
            Mnist32x32_4L_4in16Sparse(std::string weightsFileTemplate, std::string biasesFileTemplate);
            ~Mnist32x32_4L_4in16Sparse() = default;

            void predict(Matrix::Dense &input, Matrix::Dense &output);
            Matrix::Dense predict(Matrix::Dense &input);
    };

    template <uint8_t K, uint8_t N, uint8_t numberOfThreads>
    class Mnist32x32_4L_KinMSparse
    {
        private:
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> _W0;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> _W1;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> _W2;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> _W3;
            Matrix::Dense _W4;

            Matrix::Dense _B0;
            Matrix::Dense _B1;
            Matrix::Dense _B2;
            Matrix::Dense _B3;
            Matrix::Dense _B4;

            Matrix::Dense _input;
            Matrix::Dense _outputSample;
            Matrix::Dense _tmp1Sample;
            Matrix::Dense _tmp2Sample;
            Matrix::Dense _outputMatrix;
            Matrix::Dense _tmp1Matrix;
            Matrix::Dense _tmp2Matrix;

            std::counting_semaphore<numberOfThreads - 1> _semaphore;
            std::barrier<> _barrier;
            std::thread _threads[numberOfThreads - 1];
            bool _run;
            bool _predictRow;

            void predictThread(uint8_t threadId)
            {
                using namespace Matrix;

                _semaphore.acquire();
                while (_run)
                {
                    if (_predictRow)
                    {
                        _W0.template dotAddActivateRowThread<ReLU>(_input, _B0, _tmp1Sample, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W1.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B1, _tmp2Sample, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W2.template dotAddActivateRowThread<ReLU>(_tmp2Sample, _B2, _tmp1Sample, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W3.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B3, _tmp2Sample, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();

                        _W4.dotAddActivateRowThread(_tmp2Sample, _B4, _outputSample, identity, numberOfThreads, threadId);
                        _barrier.arrive_and_wait();
                    }
                    else
                    {
                        _W0.template dotAddActivateColumnThread<ReLU>(_input, _B0, _tmp1Matrix, numberOfThreads, threadId);
                        _W1.template dotAddActivateColumnThread<ReLU>(_tmp1Matrix, _B1, _tmp2Matrix, numberOfThreads, threadId);
                        _W2.template dotAddActivateColumnThread<ReLU>(_tmp2Matrix, _B2, _tmp1Matrix, numberOfThreads, threadId);
                        _W3.template dotAddActivateColumnThread<ReLU>(_tmp1Matrix, _B3, _tmp2Matrix, numberOfThreads, threadId);
                        _W4.dotAddActivateColumnThread(_tmp2Matrix, _B4, _outputMatrix, identity, numberOfThreads, threadId);        
                        _barrier.arrive_and_wait();
                    }

                    _semaphore.acquire();
                }
            }

            void startThreads()
            {
                using namespace std;

                for (uint8_t i = 1; i < numberOfThreads; i++)
                {
                    _threads[i - 1] = thread(&Mnist32x32_4L_KinMSparse::predictThread, this, i);
                }
            }

        public:
            Mnist32x32_4L_KinMSparse() : _semaphore(0), _barrier(numberOfThreads), _run(true), _predictRow(true)
            {
                startThreads();
            }

            Mnist32x32_4L_KinMSparse(std::string weightsFileTemplate, std::string biasesFileTemplate) :
                _W0{weightsFileTemplate + "l0.csv"},
                _W1{weightsFileTemplate + "l1.csv"},
                _W2{weightsFileTemplate + "l2.csv"},
                _W3{weightsFileTemplate + "l3.csv"},
                _W4{weightsFileTemplate + "l4.csv", Matrix::ROW_MAJOR},
                _B0{biasesFileTemplate + "l0.csv", Matrix::COLUMN_MAJOR},
                _B1{biasesFileTemplate + "l1.csv", Matrix::COLUMN_MAJOR},
                _B2{biasesFileTemplate + "l2.csv", Matrix::COLUMN_MAJOR},
                _B3{biasesFileTemplate + "l3.csv", Matrix::COLUMN_MAJOR},
                _B4{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR},
                _outputSample{10, 1, Matrix::COLUMN_MAJOR},
                _outputMatrix{10, 10'000, Matrix::COLUMN_MAJOR},
                _tmp1Sample{1024, 1, Matrix::COLUMN_MAJOR},
                _tmp2Sample{1024, 1, Matrix::COLUMN_MAJOR},
                _tmp1Matrix{1024, 10'000, Matrix::COLUMN_MAJOR},
                _tmp2Matrix{1024, 10'000, Matrix::COLUMN_MAJOR},
                _semaphore(0), _barrier(numberOfThreads), _run(true), _predictRow(true)
            { 
                startThreads();
            }

            ~Mnist32x32_4L_KinMSparse()
            {
                _input.clear();
                _run = false;
                _semaphore.release(numberOfThreads - 1);
                for (uint8_t i = 0; i < numberOfThreads - 1; i++)
                {
                    _threads[i].join();
                }
            }

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate)
            {
                using namespace Matrix;

                _W0 = BlockKinNSparse<K, N, 1024, 1024, ROW_MAJOR>{weightsFileTemplate + "l0.csv"};
                _W1 = BlockKinNSparse<K, N, 1024, 1024, ROW_MAJOR>{weightsFileTemplate + "l1.csv"};
                _W2 = BlockKinNSparse<K, N, 1024, 1024, ROW_MAJOR>{weightsFileTemplate + "l2.csv"};
                _W3 = BlockKinNSparse<K, N, 1024, 1024, ROW_MAJOR>{weightsFileTemplate + "l3.csv"};
                _W4 = Dense{weightsFileTemplate + "l4.csv", ROW_MAJOR};

                _B0 = Dense{biasesFileTemplate + "l0.csv", COLUMN_MAJOR};
                _B1 = Dense{biasesFileTemplate + "l1.csv", COLUMN_MAJOR};
                _B2 = Dense{biasesFileTemplate + "l2.csv", COLUMN_MAJOR};
                _B3 = Dense{biasesFileTemplate + "l3.csv", COLUMN_MAJOR};
                _B4 = Dense{biasesFileTemplate + "l4.csv", COLUMN_MAJOR};

                _outputSample = Dense{10, 1, COLUMN_MAJOR};
                _outputMatrix = Dense{10, 10'000, COLUMN_MAJOR};
                _tmp1Sample = Dense{1024, 1, COLUMN_MAJOR};
                _tmp2Sample = Dense{1024, 1, COLUMN_MAJOR};
                _tmp1Matrix = Dense{1024, 10'000, COLUMN_MAJOR};
                _tmp2Matrix = Dense{1024, 10'000, COLUMN_MAJOR};
            }

            uint8_t predictRawSample(float *input)
            {
                using namespace Matrix;

                _input.setFloatMatrix(input, 1024, 1, COLUMN_MAJOR);
                _W0.template dotAddActivateRowThread<ReLU>(_input, _B0, _tmp1Sample);
                _W1.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B1, _tmp2Sample);
                _W2.template dotAddActivateRowThread<ReLU>(_tmp2Sample, _B2,_tmp1Sample);
                _W3.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B3, _tmp2Sample);

                _W4.dot(_tmp2Sample, _outputSample);
                _outputSample.add(_B4);

                return _outputSample.argmax();
            }

            Matrix::Dense predictMatrix(Matrix::Dense input)
            {
                using namespace Matrix;

                _W0.template dotAddActivateRowThread<ReLU>(input, _B0, _tmp1Matrix);
                _W1.template dotAddActivateRowThread<ReLU>(_tmp1Matrix, _B1, _tmp2Matrix);
                _W2.template dotAddActivateRowThread<ReLU>(_tmp2Matrix, _B2,_tmp1Matrix);
                _W3.template dotAddActivateRowThread<ReLU>(_tmp1Matrix, _B3, _tmp2Matrix);

                _W4.dot(_tmp2Matrix, _outputMatrix);
                _outputMatrix.add(_B4);

                return _outputMatrix.argmax(0);
            }

            uint8_t predictThreadsRawSample(float *input)
            {
                using namespace Matrix;

                _predictRow = true;
                _input.setFloatMatrix(input, 1024, 1, COLUMN_MAJOR);
                _semaphore.release(numberOfThreads - 1);
                
                _W0.template dotAddActivateRowThread<ReLU>(_input, _B0, _tmp1Sample, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W1.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B1, _tmp2Sample, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W2.template dotAddActivateRowThread<ReLU>(_tmp2Sample, _B2, _tmp1Sample, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W3.template dotAddActivateRowThread<ReLU>(_tmp1Sample, _B3, _tmp2Sample, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W4.dotAddActivateRowThread(_tmp2Sample, _B4, _outputSample, identity, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                return _outputSample.argmax();
            }

            Matrix::Dense predictThreadsMatrix(Matrix::Dense input)
            {
                using namespace Matrix;

                _predictRow = false;
                _input.setFloatMatrix(input.getData(), input.getRows(), input.getColumns(), COLUMN_MAJOR);
                _semaphore.release(numberOfThreads - 1);

                _W0.template dotAddActivateColumnThread<ReLU>(_input, _B0, _tmp1Matrix, numberOfThreads, 0);
                _W1.template dotAddActivateColumnThread<ReLU>(_tmp1Matrix, _B1, _tmp2Matrix, numberOfThreads, 0);
                _W2.template dotAddActivateColumnThread<ReLU>(_tmp2Matrix, _B2, _tmp1Matrix, numberOfThreads, 0);
                _W3.template dotAddActivateColumnThread<ReLU>(_tmp1Matrix, _B3, _tmp2Matrix, numberOfThreads, 0);
                _W4.dotAddActivateColumnThread(_tmp2Matrix, _B4, _outputMatrix, identity, numberOfThreads, 0);

                _barrier.arrive_and_wait();

                return _outputMatrix.argmax(0);
            }
    };
}

#endif
