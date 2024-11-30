#pragma once

#include "search.hpp"

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP,team);

    board.print();

    board.doMove(0, 0, 0, 1);
    std::cout << board.evaluate() << "\n";
    board.doMove(0, 9, 0, 8);
    std::cout << board.evaluate() << "\n";

    Search ss;
    Node n = ss.searchMain(board, 1, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
}
