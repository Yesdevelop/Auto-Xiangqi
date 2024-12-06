#pragma once

#include "moves.hpp"
#include "evaluate.hpp"
#include "heuristic.hpp"
#include <cassert>
#include <windows.h>

/// @brief 节点对象，存储分数 + 着法
class Node
{
public:
    Node(Move move, int score) : move(move), score(score) {}
    Move move{};
    int score = 0;
};

class Search
{
public:
    void searchInit()
    {
        rootMoves.resize(0);
        this->historyCache->init();
    }
    void searchStep(Move &bestMove)
    {
        for (auto &move : rootMoves)
        {
            if (bestMove == move)
            {
                move.val = INF;
            }
            else
            {
                move.val--;
            }
        }
    }
    void searchIterate()
    {
        std::sort(rootMoves.begin(), rootMoves.end(), vlMoveCompare);
    }

public:
    Node searchMain(Board &board, int maxDepth, int maxTime);
    Node searchRoot(Board &board, int depth);
    int searchPV(Board &board, int depth, int alpha, int beta);
    int searchCut(Board &board, int depth, int beta);

public:
    static bool vlMoveCompare(Move &first, Move &second)
    {
        return first.val > second.val;
    }

public:
    HistoryHeuristic *historyCache = new HistoryHeuristic();
    MOVES rootMoves;
};

/// @brief 迭代加深
/// @param board
/// @param maxDepth
/// @param maxTime
/// @return
Node Search::searchMain(Board &board, int maxDepth, int maxTime = 3)
{
    searchInit();
    this->rootMoves = Moves::getMoves(board);
    Node bestNode = Node(Move(), 0);
    clock_t start = clock();
    for (int depth = 1; depth <= maxDepth; depth++)
    {
        bestNode = searchRoot(board, depth);
        //std::cout << depth << std::endl;

        if (clock() - start >= maxTime * 1000 / 3)
        {
            break;
        }

    }
    return bestNode;
}

/// @brief 根节点搜索
/// @param board
/// @param depth
/// @return
Node Search::searchRoot(Board &board, int depth)
{
    Move *pBestMove = nullptr;
    int vl = -INF;
    int vlBest = -INF;
    for (auto &move : rootMoves)
    {
        Piece eaten = board.doMove(move);
        if (vlBest == -INF)
        {
            vl = -searchPV(board, depth - 1, -INF, -vlBest);
        }
        else
        {
            vl = -searchCut(board, depth - 1, -vlBest);
            if (vl > vlBest)
            {
                vl = -searchPV(board, depth - 1, -INF, -vlBest);
            }
        }
        board.undoMove(move, eaten);

        //std::cout << vl << std::endl;
        if (vl > vlBest)
        {
            vlBest = vl;
            pBestMove = &move;
            searchStep(move);
        }
    }

    if (!pBestMove)
    {
        vlBest += depth;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
    }
    searchIterate();
    return Node(!pBestMove ? Move{} : *pBestMove, vlBest);
}

/// @brief PV搜索
/// @param board
/// @param depth
/// @param alpha
/// @param beta
/// @return
int Search::searchPV(Board &board, int depth, int alpha, int beta)
{
    if (depth <= 0)
    {
        return board.evaluate();
    }
    MOVES availableMoves = Moves::getMoves(board);
    this->historyCache->sort(availableMoves);
    Move *pBestMove = nullptr;
    int vl = -INF;
    int vlBest = -INF;
    for (auto &move : availableMoves)
    {
        Piece eaten = board.doMove(move);
        if (vlBest == -INF)
        {
            vl = -searchPV(board, depth - 1, -beta, -alpha);
        }
        else
        {
            vl = -searchCut(board, depth - 1, -alpha);
            if (vl > alpha && vl < beta)
            {
                vl = -searchPV(board, depth - 1, -beta, -alpha);
            }
        }
        board.undoMove(move, eaten);

        if (vl > vlBest)
        {
            vlBest = vl;
            pBestMove = &move;
            if (vl >= beta)
            {
                break;
            }
            if (vl > alpha)
            {
                alpha = vl;
            }
        }
    }

    if (!pBestMove)
    {
        vlBest += depth;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
    }
    return vlBest;
}

int Search::searchCut(Board &board, int depth, int beta)
{
    if (depth <= 0)
    {
        return board.evaluate();
    }
    MOVES availableMoves = Moves::getMoves(board);
    this->historyCache->sort(availableMoves);
    Move *pBestMove = nullptr;
    int vlBest = -INF;
    for (auto &move : availableMoves)
    {
        Piece eaten = board.doMove(move);
        int vl = -searchCut(board, depth - 1, -beta + 1);
        board.undoMove(move, eaten);

        if (vl > vlBest)
        {
            vlBest = vl;
            pBestMove = &move;
            if (vl >= beta)
            {
                break;
            }
        }
    }

    if (!pBestMove)
    {
        vlBest += depth;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
    }
    return vlBest;
}
