#pragma once

#include "search.hpp"

void standardTest(TEAM team = BLACK)
{
    Board board = Board(DEFAULT_MAP,team);

    board.print();
    std::cout<<board.evaluate()<<std::endl;
    std::cout<<std::endl;
//
//    Piece ep = board.doMove(Move(1,2,1,9));
//    board.print();
//    std::cout<<board.evaluate()<<std::endl;
//    board.undoMove(Move(1,2,1,9),ep);
//    board.print();
//    std::cout<<board.evaluate()<<std::endl;

    Search ss;
    Node n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
    n = ss.searchMain(board, 6, 1);
    board.doMove(n.move);
    board.print();
    std::cout << "score: " << n.score << std::endl;
}
