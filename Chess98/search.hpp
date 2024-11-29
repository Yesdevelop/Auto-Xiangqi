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

        std::cout << depth << std::endl;

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
        return Evaluate::evaluate(board);
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
        return Evaluate::evaluate(board);
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

/// @brief 判断当前一方是否被将军
/// @param board
/// @return
bool isChecked(Board &board)
{
    Piece *king = board.team == RED ? board.pieceRedKing : board.pieceRedKing;
    // 判断敌方的兵是否在附近
    bool c1 = abs(board.pieceidOn(king->x + 1, king->y)) == R_PAWN;
    bool c2 = abs(board.pieceidOn(king->x - 1, king->y)) == R_PAWN;
    bool c3 = abs(board.pieceidOn(king->x, king->y + 1)) == R_PAWN;
    // 判断敌方的马是否在附近
    auto piece1 = board.findPieceByIndex(board.pieceidOn(king->x + 2, king->y + 1));
    auto piece2 = board.findPieceByIndex(board.pieceidOn(king->x - 2, king->y + 1));
    auto piece3 = board.findPieceByIndex(board.pieceidOn(king->x + 2, king->y - 1));
    auto piece4 = board.findPieceByIndex(board.pieceidOn(king->x - 2, king->y - 1));
    auto piece5 = board.findPieceByIndex(board.pieceidOn(king->x + 1, king->y + 2));
    auto piece6 = board.findPieceByIndex(board.pieceidOn(king->x + 1, king->y - 2));
    auto piece7 = board.findPieceByIndex(board.pieceidOn(king->x - 1, king->y + 2));
    auto piece8 = board.findPieceByIndex(board.pieceidOn(king->x - 1, king->y - 2));
    bool c4 = abs(piece1.pieceid) == R_KNIGHT && piece1.getTeam() != board.team;
    bool c5 = abs(piece2.pieceid) == R_KNIGHT && piece2.getTeam() != board.team;
    bool c6 = abs(piece3.pieceid) == R_KNIGHT && piece3.getTeam() != board.team;
    bool c7 = abs(piece4.pieceid) == R_KNIGHT && piece4.getTeam() != board.team;
    bool c8 = abs(piece5.pieceid) == R_KNIGHT && piece5.getTeam() != board.team;
    bool c9 = abs(piece6.pieceid) == R_KNIGHT && piece6.getTeam() != board.team;
    bool c10 = abs(piece7.pieceid) == R_KNIGHT && piece7.getTeam() != board.team;
    bool c11 = abs(piece8.pieceid) == R_KNIGHT && piece8.getTeam() != board.team;
    // 判断是否被将军
    bool condition = c1 || c2 || c3 || c4 || c5 || c6 || c7 || c8 || c9 || c10 || c11;
    if (condition == true)
        return true;

    // 白脸将、车、炮
    bool barrierDetected = false;
    for (int x = king->x + 1; x < 8; x++)
    {
        PIECEID pieceid = board.pieceidOn(x, king->y);
        TEAM team = board.teamOn(x, king->y);
        if (abs(pieceid) == R_ROOK &&
            abs(pieceid) == R_KING &&
            team != board.team &&
            barrierDetected == false)
        {
            return true;
        }
        else if (abs(pieceid) == R_CANNON &&
                 team != board.team &&
                 barrierDetected == true)
        {
            return true;
        }
        else if (pieceid != 0)
        {
            barrierDetected = true;
        }
    }
    barrierDetected = false;
    for (int x = king->x - 1; x < 8; x--)
    {
        PIECEID pieceid = board.pieceidOn(x, king->y);
        TEAM team = board.teamOn(x, king->y);
        if (abs(pieceid) == R_ROOK &&
            abs(pieceid) == R_KING &&
            team != board.team &&
            barrierDetected == false)
        {
            return true;
        }
        else if (abs(pieceid) == R_CANNON &&
                 team != board.team &&
                 barrierDetected == true)
        {
            return true;
        }
        else if (pieceid != 0)
        {
            barrierDetected = true;
        }
    }
    barrierDetected = false;
    if (board.team == RED)
    {
        for (int y = king->y + 1; y < 9; y++)
        {
            PIECEID pieceid = board.pieceidOn(king->x, y);
            TEAM team = board.teamOn(king->x, y);
            if (abs(pieceid) == R_ROOK &&
                abs(pieceid) == R_KING &&
                team != board.team &&
                barrierDetected == false)
            {
                return true;
            }
            else if (abs(pieceid) == R_CANNON &&
                     team != board.team &&
                     barrierDetected == true)
            {
                return true;
            }
            else if (pieceid != 0)
            {
                barrierDetected = true;
            }
        }
    }
    else {
        for (int y = king->y - 1; y >= 0; y--)
        {
            PIECEID pieceid = board.pieceidOn(king->x, y);
            TEAM team = board.teamOn(king->x, y);
            if (abs(pieceid) == R_ROOK &&
                abs(pieceid) == R_KING &&
                team != board.team &&
                barrierDetected == false)
            {
                return true;
            }
            else if (abs(pieceid) == R_CANNON &&
                     team != board.team &&
                     barrierDetected == true)
            {
                return true;
            }
            else if (pieceid != 0)
            {
                barrierDetected = true;
            }
        }
    }
}
