#include "search.hpp"

int main()
{
    initZobrist();
    HistoryHeuristic::init();

    Board board{DEFAULT_MAP};

    MOVES a = Moves::getMovesOf(board, RED);

    board.print();

    // MOVES moves{ Move{0, 0, 0, 1}, Move{0, 0, 0, 2}, Move{4, 0, 4, 1} };
    // HistoryHeuristic::sort(moves);

    Search::alphabeta(board, 0, true, -100000, 100000);
    __count__ = 0;
    std::cout << "a";
    Search::alphabeta(board, 1, true, -100000, 100000);
    __count__ = 0;
    std::cout << "a";
    Search::alphabeta(board, 2, true, -100000, 100000);
    __count__ = 0;
    std::cout << "a";
    Search::alphabeta(board, 3, true, -100000, 100000);
    __count__ = 0;
    std::cout << "a";
    Search::alphabeta(board, 4, true, -100000, 100000);
    __count__ = 0;
    std::cout << "a";
    Search::alphabeta(board, 5, true, -100000, 100000);
    std::cout << "a";
    __count__ = 0;
    Search::alphabeta(board, 6, true, -100000, 100000);
    std::cout << "a";
    std::cout << __count__ << std::endl;
    system("pause");

    return 0;
}
