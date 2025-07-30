#include "test.hpp"

int main()
{
#ifndef NNUE
    std::cout << "You're runing in UI mode, and open ui.html to play xiangqi now!\n"
              << std::endl;
    testByUI();
#else
    std::cout << "You're running in NNUE file generate mode. It could be disabled by remove 'define NNUE' in nnuefile.hpp!\n"
              << std::endl;
    testGenerateNNUE();
#endif

    return 0;
}
