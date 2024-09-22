#include "search.hpp"

int main()
{
    Board board{ DEFAULT_MAP };

    MOVES a = MovesGenerator::getMovesOf(board, RED);

    board.print();
    Node s = Searcher::search(board, RED, 1);

    return 0;
}
