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

        public:
            Mnist32x32_4L() = default;
            Mnist32x32_4L(std::string weightsFileTemplate, std::string biasesFileTemplate);
            ~Mnist32x32_4L() = default;

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate);

            void predict(Matrix::Dense &input, Matrix::Dense &output);
            Matrix::Dense predict(Matrix::Dense &input);

            void predictOptimized(Matrix::Dense &input, Matrix::Dense &output);
            Matrix::Dense predictOptimized(Matrix::Dense &input);
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
            Matrix::Dense _output;
            Matrix::Dense _tmp1;
            Matrix::Dense _tmp2;

            std::counting_semaphore<numberOfThreads - 1> _semaphore;
            std::barrier<> _barrier;
            std::thread _threads[numberOfThreads - 1];
            bool _run;

            static float identity(float x)
            {
                return x;
            }

            void predictThread(uint16_t threadId)
            {
                using namespace Matrix;

                _semaphore.acquire();        
                while (_run)
                {
                    _W0.dotAddActivateThread(_input, _B0, _tmp1, ReLU, numberOfThreads, threadId);
                    _barrier.arrive_and_wait();

                    _W1.dotAddActivateThread(_tmp1, _B1, _tmp2, ReLU, numberOfThreads, threadId);
                    _barrier.arrive_and_wait();

                    _W2.dotAddActivateThread(_tmp2, _B2, _tmp1, ReLU, numberOfThreads, threadId);
                    _barrier.arrive_and_wait();

                    _W3.dotAddActivateThread(_tmp1, _B3, _tmp2, ReLU, numberOfThreads, threadId);
                    _barrier.arrive_and_wait();

                    _W4.dotAddActivateThread(_tmp2, _B4, _output, identity, numberOfThreads, threadId);
                    _barrier.arrive_and_wait();

                    _semaphore.acquire();
                }
            }

        public:
            Mnist32x32_4L_Threads() : _semaphore(0), _barrier(numberOfThreads), _run(true)
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

                _W0 = Dense{weightsFileTemplate + "l0.csv", ROW_MAJOR, 0 , numberOfThreads};
                _W1 = Dense{weightsFileTemplate + "l1.csv", ROW_MAJOR, 0 , numberOfThreads};
                _W2 = Dense{weightsFileTemplate + "l2.csv", ROW_MAJOR, 0 , numberOfThreads};
                _W3 = Dense{weightsFileTemplate + "l3.csv", ROW_MAJOR, 0 , numberOfThreads};
                _W4 = Dense{weightsFileTemplate + "l4.csv", ROW_MAJOR, 0 , numberOfThreads};

                _B0 = Dense{biasesFileTemplate + "l0.csv", COLUMN_MAJOR, 0 , numberOfThreads};
                _B1 = Dense{biasesFileTemplate + "l1.csv", COLUMN_MAJOR, 0 , numberOfThreads};
                _B2 = Dense{biasesFileTemplate + "l2.csv", COLUMN_MAJOR, 0 , numberOfThreads};
                _B3 = Dense{biasesFileTemplate + "l3.csv", COLUMN_MAJOR, 0 , numberOfThreads};
                _B4 = Dense{biasesFileTemplate + "l4.csv", COLUMN_MAJOR, 0 , numberOfThreads};

                _output = Dense(_B4.getRows(), 1, COLUMN_MAJOR, numberOfThreads);
                _tmp1 = Dense(_B0.getRows(), 1, COLUMN_MAJOR, numberOfThreads);
                _tmp2 = Dense(_B0.getRows(), 1, COLUMN_MAJOR, numberOfThreads);
            }

            uint32_t predictRaw(float *input)
            {
                using namespace Matrix;

                _input.setFloatMatrix(input, _W0.getColumns(), 1, COLUMN_MAJOR);
                _semaphore.release(numberOfThreads - 1);

                _W0.dotAddActivateThread(_input, _B0, _tmp1, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W1.dotAddActivateThread(_tmp1, _B1, _tmp2, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W2.dotAddActivateThread(_tmp2, _B2, _tmp1, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W3.dotAddActivateThread(_tmp1, _B3, _tmp2, ReLU, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                _W4.dotAddActivateThread(_tmp2, _B4, _output, identity, numberOfThreads, 0);
                _barrier.arrive_and_wait();

                return _output.argmax();
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

    template <uint8_t K, uint8_t N>
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

            template <float (*activationFunction)(float)>
            void predictActivationThread(Matrix::Dense &input, Matrix::Dense &tmp, uint8_t numberOfThreads, uint8_t threadId, 
                                         std::barrier<> &syncBarrier)
            {
                _W0.template dotAddActivate<activationFunction>(input, _B0, tmp, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                _W1.template dotAddActivate<activationFunction>(tmp, _B1, input, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                _W2.template dotAddActivate<activationFunction>(input, _B2, tmp, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                _W3.template dotAddActivate<activationFunction>(tmp, _B3, input, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();
            }

        public:
            Mnist32x32_4L_KinMSparse() {};
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
                _B4{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR} 
            { }

            ~Mnist32x32_4L_KinMSparse() = default;

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate)
            {
                _W0 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l0.csv"};
                _W1 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l1.csv"};
                _W2 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l2.csv"};
                _W3 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l3.csv"};
                _W4 = Matrix::Dense{weightsFileTemplate + "l4.csv", Matrix::ROW_MAJOR};

                _B0 = Matrix::Dense{biasesFileTemplate + "l0.csv", Matrix::COLUMN_MAJOR};
                _B1 = Matrix::Dense{biasesFileTemplate + "l1.csv", Matrix::COLUMN_MAJOR};
                _B2 = Matrix::Dense{biasesFileTemplate + "l2.csv", Matrix::COLUMN_MAJOR};
                _B3 = Matrix::Dense{biasesFileTemplate + "l3.csv", Matrix::COLUMN_MAJOR};
                _B4 = Matrix::Dense{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR};
            }

            void predict(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp = _W0.dot(input);
                tmp.add(_B0);
                tmp.ReLU();
                
                _W1.dot(tmp, input);
                input.add(_B1);
                input.ReLU();
            
                _W2.dot(input, tmp);
                tmp.add(_B2);
                tmp.ReLU();
            
                _W3.dot(tmp, input);
                input.add(_B3);
                input.ReLU();
            
                _W4.dot(input, output);
                output.add(_B4);
            }

            Matrix::Dense predict(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predict(input, output);
                return output;
            }

            void predictThreads(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp = _W0.dotThreads(input);
                tmp.add(_B0);
                tmp.ReLU();
                
                _W1.dotThreads(tmp, input);
                input.add(_B1);
                input.ReLU();
            
                _W2.dotThreads(input, tmp);
                tmp.add(_B2);
                tmp.ReLU();
            
                _W3.dotThreads(tmp, input);
                input.add(_B3);
                input.ReLU();
            
                _W4.dot(input, output);
                output.add(_B4);
            }

            Matrix::Dense predictThreads(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predictThreads(input, output);
                return output;
            }
 
            void predictOptimized(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp(_W0.getRows(), input.getColumns(), COLUMN_MAJOR);

                _W0.template dotAddActivate<ReLU>(input, _B0, tmp);                              
                _W1.template dotAddActivate<ReLU>(tmp, _B1, input);
                _W2.template dotAddActivate<ReLU>(input, _B2, tmp);
                _W3.template dotAddActivate<ReLU>(tmp, _B3, input);

                _W4.dot(input, output);
                output.add(_B4);
            }

            Matrix::Dense predictOptimized(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predictOptimized(input, output);
                return output;
            }

            template <uint8_t numberOfThreads>
            void predictOptimizedThreads(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                using namespace std;

                thread threads[numberOfThreads];
                barrier syncBarrier(numberOfThreads);
                
                Dense tmp(_W0.getRows(), input.getColumns(), COLUMN_MAJOR);

                for (uint8_t i = 0; i < numberOfThreads; i++)
                {
                    threads[i] = thread(&Mnist32x32_4L_KinMSparse::predictActivationThread<ReLU>, this, ref(input), ref(tmp), numberOfThreads, i, ref(syncBarrier));
                }

                for (uint8_t i = 0; i < numberOfThreads; i++)
                {
                    threads[i].join();
                }

                _W4.dot(input, output);
                output.add(_B4);
            }

            template <uint8_t numberOfThreads>
            Matrix::Dense predictOptimizedThreads(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predictOptimizedThreads<numberOfThreads>(input, output);
                return output;
            }
    };
}

#endif
