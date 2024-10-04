#include "search.hpp"

int main()
{
    initZobrist();
    HistoryHeuristic::init();

    Board board{DEFAULT_MAP};

    MOVES a = Moves::getMovesOf(board, RED);

    board.print();

    Node s = Search::alphabeta(board, 3, true, -100000, 100000);

    system("pause");

    return 0;
}
