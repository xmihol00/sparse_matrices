#ifndef MODELS_H
#define MODELS_H

#include <thread>
#include <barrier>
#include <functional>

#include "dense.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"
#include "activations.h"

namespace Models
{
    class Mnist32x32_4L
    {
        private:
            Matrix::Dense W0;
            Matrix::Dense W1;
            Matrix::Dense W2;
            Matrix::Dense W3;
            Matrix::Dense W4;

            Matrix::Dense B0;
            Matrix::Dense B1;
            Matrix::Dense B2;
            Matrix::Dense B3;
            Matrix::Dense B4;

        public:
            Mnist32x32_4L(std::string weightsFileTemplate, std::string biasesFileTemplate);
            ~Mnist32x32_4L() = default;

            void predict(Matrix::Dense &input, Matrix::Dense &output);
            Matrix::Dense predict(Matrix::Dense &input);
    };

    class Mnist32x32_4L_4in16Sparse
    {
        private:
            Matrix::Block4in16Sparse W0;
            Matrix::Block4in16Sparse W1;
            Matrix::Block4in16Sparse W2;
            Matrix::Block4in16Sparse W3;
            Matrix::Dense W4;

            Matrix::Dense B0;
            Matrix::Dense B1;
            Matrix::Dense B2;
            Matrix::Dense B3;
            Matrix::Dense B4;

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
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> W0;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> W1;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> W2;
            Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR> W3;
            Matrix::Dense W4;

            Matrix::Dense B0;
            Matrix::Dense B1;
            Matrix::Dense B2;
            Matrix::Dense B3;
            Matrix::Dense B4;

            template <float (*activationFunction)(float)>
            void predictActivationThread(Matrix::Dense &input, Matrix::Dense &tmp, uint8_t numberOfThreads, uint8_t threadId, 
                                         std::barrier<> &syncBarrier)
            {
                W0.template dotAddActivate<activationFunction>(input, B0, tmp, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                W1.template dotAddActivate<activationFunction>(tmp, B1, input, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                W2.template dotAddActivate<activationFunction>(input, B2, tmp, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();

                W3.template dotAddActivate<activationFunction>(tmp, B3, input, numberOfThreads, threadId);
                syncBarrier.arrive_and_wait();
            }

        public:
            Mnist32x32_4L_KinMSparse() {};
            Mnist32x32_4L_KinMSparse(std::string weightsFileTemplate, std::string biasesFileTemplate) : 
                W0{weightsFileTemplate + "l0.csv"},
                W1{weightsFileTemplate + "l1.csv"},
                W2{weightsFileTemplate + "l2.csv"},
                W3{weightsFileTemplate + "l3.csv"},
                W4{weightsFileTemplate + "l4.csv", Matrix::ROW_MAJOR},
                B0{biasesFileTemplate + "l0.csv", Matrix::COLUMN_MAJOR},
                B1{biasesFileTemplate + "l1.csv", Matrix::COLUMN_MAJOR},
                B2{biasesFileTemplate + "l2.csv", Matrix::COLUMN_MAJOR},
                B3{biasesFileTemplate + "l3.csv", Matrix::COLUMN_MAJOR},
                B4{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR} 
            { }

            ~Mnist32x32_4L_KinMSparse() = default;

            void load(std::string weightsFileTemplate, std::string biasesFileTemplate)
            {
                W0 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l0.csv"};
                W1 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l1.csv"};
                W2 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l2.csv"};
                W3 = Matrix::BlockKinNSparse<K, N, 1024, 1024, Matrix::ROW_MAJOR>{weightsFileTemplate + "l3.csv"};
                W4 = Matrix::Dense{weightsFileTemplate + "l4.csv", Matrix::ROW_MAJOR};

                B0 = Matrix::Dense{biasesFileTemplate + "l0.csv", Matrix::COLUMN_MAJOR};
                B1 = Matrix::Dense{biasesFileTemplate + "l1.csv", Matrix::COLUMN_MAJOR};
                B2 = Matrix::Dense{biasesFileTemplate + "l2.csv", Matrix::COLUMN_MAJOR};
                B3 = Matrix::Dense{biasesFileTemplate + "l3.csv", Matrix::COLUMN_MAJOR};
                B4 = Matrix::Dense{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR};
            }

            void predict(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp = W0.dot(input);
                tmp.add(B0);
                tmp.ReLU();
                
                W1.dot(tmp, input);
                input.add(B1);
                input.ReLU();
            
                W2.dot(input, tmp);
                tmp.add(B2);
                tmp.ReLU();
            
                W3.dot(tmp, input);
                input.add(B3);
                input.ReLU();
            
                W4.dot(input, output);
                output.add(B4);
            }

            Matrix::Dense predict(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predict(input, output);
                return output;
            }

            void predictThreads(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp = W0.dotThreads(input);
                tmp.add(B0);
                tmp.ReLU();
                
                W1.dotThreads(tmp, input);
                input.add(B1);
                input.ReLU();
            
                W2.dotThreads(input, tmp);
                tmp.add(B2);
                tmp.ReLU();
            
                W3.dotThreads(tmp, input);
                input.add(B3);
                input.ReLU();
            
                W4.dot(input, output);
                output.add(B4);
            }

            Matrix::Dense predictThreads(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predictThreads(input, output);
                return output;
            }
 
            void predictOptimized(Matrix::Dense &input, Matrix::Dense &output)
            {
                using namespace Matrix;
                
                Dense tmp(W0.getRows(), input.getColumns(), COLUMN_MAJOR);

                W0.template dotAddActivate<ReLU>(input, B0, tmp);                              
                W1.template dotAddActivate<ReLU>(tmp, B1, input);
                W2.template dotAddActivate<ReLU>(input, B2, tmp);
                W3.template dotAddActivate<ReLU>(tmp, B3, input);

                W4.dot(input, output);
                output.add(B4);
            }

            Matrix::Dense predictOptimized(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(B4.getRows(), input.getColumns(), COLUMN_MAJOR);
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
                
                Dense tmp(W0.getRows(), input.getColumns(), COLUMN_MAJOR);

                for (uint8_t i = 0; i < numberOfThreads; i++)
                {
                    threads[i] = thread(&Mnist32x32_4L_KinMSparse::predictActivationThread<ReLU>, this, ref(input), ref(tmp), numberOfThreads, i, ref(syncBarrier));
                }

                for (uint8_t i = 0; i < numberOfThreads; i++)
                {
                    threads[i].join();
                }

                W4.dot(input, output);
                output.add(B4);
            }

            template <uint8_t numberOfThreads>
            Matrix::Dense predictOptimizedThreads(Matrix::Dense &input)
            {
                using namespace Matrix;

                Dense output(B4.getRows(), input.getColumns(), COLUMN_MAJOR);
                predictOptimizedThreads<numberOfThreads>(input, output);
                return output;
            }
    };
}

#endif
