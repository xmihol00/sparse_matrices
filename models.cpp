
#include "models.h"

using namespace std;
using namespace Matrix;
using namespace Models;

Mnist32x32_4L::Mnist32x32_4L(string weightsFileTemplate, string biasesFileTemplate)
    : W0{weightsFileTemplate + "l0.csv", ROW_MAJOR},
      W1{weightsFileTemplate + "l1.csv", ROW_MAJOR},
      W2{weightsFileTemplate + "l2.csv", ROW_MAJOR},
      W3{weightsFileTemplate + "l3.csv", ROW_MAJOR},
      W4{weightsFileTemplate + "l4.csv", ROW_MAJOR},
      B0{biasesFileTemplate + "l0.csv", COLUMN_MAJOR},
      B1{biasesFileTemplate + "l1.csv", COLUMN_MAJOR},
      B2{biasesFileTemplate + "l2.csv", COLUMN_MAJOR},
      B3{biasesFileTemplate + "l3.csv", COLUMN_MAJOR},
      B4{biasesFileTemplate + "l4.csv", COLUMN_MAJOR}
{
}

void Mnist32x32_4L::predict(Dense &input, Dense &output)
{
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

Dense Mnist32x32_4L::predict(Dense &input)
{
    Dense output(B4.getRows(), input.getColumns(), COLUMN_MAJOR);
    predict(input, output);
    return output;
}
