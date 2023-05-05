
#include "models.h"

using namespace std;
using namespace Matrix;
using namespace Models;

Mnist32x32_4L::Mnist32x32_4L(string weightsFileTemplate, string biasesFileTemplate)
    : _W0{weightsFileTemplate + "l0.csv", ROW_MAJOR},
      _W1{weightsFileTemplate + "l1.csv", ROW_MAJOR},
      _W2{weightsFileTemplate + "l2.csv", ROW_MAJOR},
      _W3{weightsFileTemplate + "l3.csv", ROW_MAJOR},
      _W4{weightsFileTemplate + "l4.csv", ROW_MAJOR},
      _B0{biasesFileTemplate + "l0.csv", COLUMN_MAJOR},
      _B1{biasesFileTemplate + "l1.csv", COLUMN_MAJOR},
      _B2{biasesFileTemplate + "l2.csv", COLUMN_MAJOR},
      _B3{biasesFileTemplate + "l3.csv", COLUMN_MAJOR},
      _B4{biasesFileTemplate + "l4.csv", COLUMN_MAJOR}
{ }

void Mnist32x32_4L::load(std::string weightsFileTemplate, std::string biasesFileTemplate)
{
    _W0 = Dense{weightsFileTemplate + "l0.csv", Matrix::ROW_MAJOR};
    _W1 = Dense{weightsFileTemplate + "l1.csv", Matrix::ROW_MAJOR};
    _W2 = Dense{weightsFileTemplate + "l2.csv", Matrix::ROW_MAJOR};
    _W3 = Dense{weightsFileTemplate + "l3.csv", Matrix::ROW_MAJOR};
    _W4 = Dense{weightsFileTemplate + "l4.csv", Matrix::ROW_MAJOR};

    _B0 = Dense{biasesFileTemplate + "l0.csv", Matrix::COLUMN_MAJOR};
    _B1 = Dense{biasesFileTemplate + "l1.csv", Matrix::COLUMN_MAJOR};
    _B2 = Dense{biasesFileTemplate + "l2.csv", Matrix::COLUMN_MAJOR};
    _B3 = Dense{biasesFileTemplate + "l3.csv", Matrix::COLUMN_MAJOR};
    _B4 = Dense{biasesFileTemplate + "l4.csv", Matrix::COLUMN_MAJOR};
}

void Mnist32x32_4L::predict(Dense &input, Dense &output)
{
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

Dense Mnist32x32_4L::predict(Dense &input)
{
    Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
    predict(input, output);
    return output;
}

void Mnist32x32_4L::predictOptimized(Matrix::Dense &input, Matrix::Dense &output)
{
    Dense tmp(_W0.getRows(), input.getColumns(), COLUMN_MAJOR);

    _W0.dotAddActivate(input, _B0, tmp, ReLU);
    _W1.dotAddActivate(tmp, _B1, input, ReLU);
    _W2.dotAddActivate(input, _B2, tmp, ReLU);
    _W3.dotAddActivate(tmp, _B3, input, ReLU);

    _W4.dotNEON(input, output);
    output.add(_B4);
}

Dense Mnist32x32_4L::predictOptimized(Matrix::Dense &input)
{
    Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
    predictOptimized(input, output);
    return output;
}

Mnist32x32_4L_4in16Sparse::Mnist32x32_4L_4in16Sparse(string weightsFileTemplate, string biasesFileTemplate)
: _W0{weightsFileTemplate + "l0.csv", ROW_MAJOR},
      _W1{weightsFileTemplate + "l1.csv", ROW_MAJOR},
      _W2{weightsFileTemplate + "l2.csv", ROW_MAJOR},
      _W3{weightsFileTemplate + "l3.csv", ROW_MAJOR},
      _W4{weightsFileTemplate + "l4.csv", ROW_MAJOR},
      _B0{biasesFileTemplate + "l0.csv", COLUMN_MAJOR},
      _B1{biasesFileTemplate + "l1.csv", COLUMN_MAJOR},
      _B2{biasesFileTemplate + "l2.csv", COLUMN_MAJOR},
      _B3{biasesFileTemplate + "l3.csv", COLUMN_MAJOR},
      _B4{biasesFileTemplate + "l4.csv", COLUMN_MAJOR}
{ }

void Mnist32x32_4L_4in16Sparse::predict(Matrix::Dense &input, Matrix::Dense &output)
{
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

Matrix::Dense Mnist32x32_4L_4in16Sparse::predict(Matrix::Dense &input)
{
    Dense output(_B4.getRows(), input.getColumns(), COLUMN_MAJOR);
    predict(input, output);
    return output;
}
