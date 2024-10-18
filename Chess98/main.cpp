#include "test.hpp"

int main()
{
    initZobrist();
    HistoryHeuristic::init();

    standardTest();

    return 0;
}
