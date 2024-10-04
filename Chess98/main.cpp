#include "search.hpp"

int main()
{
    initZobrist();
    HistoryHeuristic::init();

    Board board{DEFAULT_MAP};

    MOVES a = Moves::getMovesOf(board, RED);

    board.print();

    Node s = Search::alphabeta(board, 5, true, -INF, INF);

    board.doMove(s.move);

    board.print();

    //system("pause");

    return 0;
}
