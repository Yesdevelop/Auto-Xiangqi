#pragma once

#include "search.hpp"

void standardTest(Board board = DEFAULT_MAP, TEAM team = BLACK)
{
    board.print();
    auto v = Moves::bishop(BLACK, board, 2, 9);

    std::cout << Evaluate::evaluate(board);
    Node s{Move{}, 0};

    s = Search::alphabeta(board, 4, 4, BLACK, -INF, INF);

    board.doMove(s.move);

    board.print();

    std::cout << "score: " << s.score << std::endl;

    std::cout << __count__ << std::endl;

    system("pause");

    standardTest(board, -team);
}
