#ifndef MODELS_H
#define MODELS_H

#include "dense.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"

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

        public:
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
    };
}

#endif
