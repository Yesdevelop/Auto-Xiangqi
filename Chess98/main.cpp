#include "test.hpp"

int main()
{
#ifndef NNUE
    testByUI();
#else
    testGenerateNNUE();
#endif
    return 0;
}
