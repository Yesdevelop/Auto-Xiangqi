#pragma once

#include "moves.hpp"
#include "evaluate.hpp"
#include "heuristic.hpp"

/// @brief 节点对象，存储分数 + 着法
class Node
{
public:
    Node(Move move, int score) : move(move), score(score) {}
    Move move{};
    int score = 0;
};

/// @brief 搜索
class Search
{
public:
    static Node search(Board &board, TEAM currentTeam, int time);
    static Node alphabeta(Board &board, int depth, int maxDepth, TEAM isRedGo, int alpha, int beta);
};

/// @brief 迭代加深搜索
/// @param board
/// @param currentTeam
/// @param depth
/// @return
Node Search::search(Board &board, TEAM currentTeam, int time)
{
    Node result{Move{}, 0};
    int depth = 0;

    TIME_T startTime = getCurrentTimeWithMS();
    do
    {
        result = Search::alphabeta(
            board, depth, depth,
            currentTeam == RED ? true : false,
            -100000, 100000);
        depth += 1;
    } while (getCurrentTimeWithMS() - startTime < time);

    return result;
}

int __count__ = 0;

/// @brief alphabeta搜索
/// @param depth 深度
/// @param isMax 节点类型，true为max节点，false为min节点
/// @return 节点
Node Search::alphabeta(Board &board, int depth, int maxDepth, TEAM currentTeam, int alpha, int beta)
{
    __count__++;
    if (depth <= 0)
    {
        int eval = currentTeam == RED ? Evaluate::evaluate(board) : -Evaluate::evaluate(board);
        return Node(Move(), eval);
    }
    MOVES availableMoves = Moves::getMovesOf(board, currentTeam);
    HistoryHeuristic::sort(availableMoves);

    Move *pBestMove = nullptr;
    int vlBest = -INF;
    for (auto &move : availableMoves)
    {
        Piece eaten = board.doMove(move);
        int vl = -Search::alphabeta(board, depth - 1, maxDepth, -currentTeam, -beta, -alpha).score;
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

    if (pBestMove)
    {
        Node node{*pBestMove, vlBest};
        HistoryHeuristic::add(node.move, maxDepth + 1 - depth);
        return node;
    }
    else
    {
        return Node{Move{}, currentTeam ? -INF + depth : INF - depth};
    }
}
