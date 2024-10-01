#include "search.hpp"

int main()
{
    initZobrist();
    
    Board board{ DEFAULT_MAP };

    MOVES a = MovesGenerator::getMovesOf(board, RED);

    board.print();
    Node s = Searcher::search(board, RED, 1000);
    system("pause");

    return 0;
}
