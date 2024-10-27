#pragma once

#include "search.hpp"

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP,team);
    board.print();
    Search ss;
    Node n = ss.searchMain(board,6,10);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
}
