#pragma once

#include "search.hpp"
#include "ui.hpp"

using std::cout;
using std::endl;

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP, team);

    serverInit(board);

    std::cout << board.evaluate() << "\n\n" << std::endl;

    board.print();

    Search s;
}
