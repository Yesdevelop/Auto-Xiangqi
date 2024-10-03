#include "search.hpp"

int main()
{
    initZobrist();
    HistoryHeuristic::init();

    Board board{ DEFAULT_MAP };

    MOVES a = Moves::getMovesOf(board, RED);

    board.print();
    Node s = Search::search(board, RED, 300);
    system("pause");

    return 0;
}
