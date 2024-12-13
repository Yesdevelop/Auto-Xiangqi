#pragma once

#include "search.hpp"

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP, team);

    Search s;

    while (true)
    {
        Node n = s.searchMain(board, 10, 1);
        board.doMove(n.move);
        board.print();
    }
}
