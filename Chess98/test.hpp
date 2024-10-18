#pragma once

#include "search.hpp"

void standardTest()
{
    Board board{DEFAULT_MAP};

    MOVES a = Moves::getMovesOf(board, RED);

    board.print();

    Node s{Move{}, 0};

    s = Search::alphabeta(board, 0, 0, true, -INF, INF);
    s = Search::alphabeta(board, 1, 1, true, -INF, INF);
    s = Search::alphabeta(board, 2, 2, true, -INF, INF);
    s = Search::alphabeta(board, 3, 3, true, -INF, INF);

    __count__ = 0;

    s = Search::alphabeta(board, 4, 4, true, -INF, INF);

    board.doMove(s.move);

    board.print();

    std::cout << __count__ << std::endl;

    system("pause");
}
