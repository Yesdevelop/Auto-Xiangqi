#include "search.hpp"

int main()
{
    Board board{ DEFAULT_MAP };

    MOVES a = MovesGenerator::getMovesOf(board, RED);

    board.print();
    Searcher::minmax(board, 3, true);

    system("pause");
    return 0;
}
