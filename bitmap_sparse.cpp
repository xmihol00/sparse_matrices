#include "bitmap_sparse.h"

using namespace std;
using namespace SparseMatrix;

BitmapSparse::BitmapSparse(string fileName)
{
    if (fileName.ends_with(".csv"))
    {
        parseCSV(fileName);
    }
    else if (fileName.ends_with(".bms"))
    {

    }
    else
    {
        throw invalid_argument("Unsupported file extension.");
    }

}

BitmapSparse::~BitmapSparse()
{
    delete[] _matrix;
}

void BitmapSparse::parseCSV(string fileName)
{
    cout << "parsing: " << fileName << endl;
}

void BitmapSparse::parseBMS(string fileName)
{

}
