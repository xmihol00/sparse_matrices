#ifndef MODELS_H
#define MODELS_H

#include "dense.h"
#include "block4in16_sparse.h"

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
}

#endif
