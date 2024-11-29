#pragma once

#include "search.hpp"

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP,team);

    board.print();
    Search ss;
    Node n = ss.searchMain(board,40,3);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
}
