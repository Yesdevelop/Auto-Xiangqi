#pragma once
#include "base.hpp"
#include "board.hpp"

/* ***** 历史启发 ***** */

/// @brief 历史启发
class HistoryHeuristic
{
public:
    void init();
    void sort(MOVES &moves);
    void add(Move move, int depth);

    static bool vlHisCompare(Move &first, Move &second)
    {
        return first.val > second.val;
    }

    int historyTable[90][90];
};

/// @brief 二维坐标转索引
/// @param x
/// @param y
/// @return
int toIndex(int x, int y)
{
    return 10 * x + y;
}

/// @brief 初始化
void HistoryHeuristic::init()
{
    std::memset(this->historyTable, 0, sizeof(int) * 90 * 90);
}

/// @brief 历史表排序
/// @param moves
/// @return
void HistoryHeuristic::sort(MOVES &moves)
{
    for (Move &move : moves)
    {
        int pos1 = toIndex(move.x1, move.y1);
        int pos2 = toIndex(move.x2, move.y2);
        move.val = historyTable[pos1][pos2];
    }
    std::sort(moves.begin(), moves.end(), vlHisCompare);
}

/// @brief 在历史表中增加一个历史记录
/// @param move
/// @param depth
void HistoryHeuristic::add(Move move, int depth)
{
    int pos1 = toIndex(move.x1, move.y1);
    int pos2 = toIndex(move.x2, move.y2);
    historyTable[pos1][pos2] += depth * depth;
}

/* ***** 置换表启发 ***** */

U64 zobristMap[7][2][9][10]{};

/// @brief 初始化zobrist哈希表
void initZobrist()
{
    memset(zobristMap, static_cast<U64>(0), sizeof(U64) * 7 * 2 * 9 * 10);
}

/* ***** 吃子启发 ***** */
void eatenHeuristic(Board board, MOVES& moves)
{
    MOVES eatenMoves{};
    MOVES result{};
    for (Move& move : moves)
    {
        if (board.pieceidOn(move.x2, move.y2) != 0)
        {
            eatenMoves.emplace_back(move);
            move.x1 = -1; // 标记一下move
        }
    }

    std::vector<int> moveWeights{};
    std::map<int, MOVES> orderMap{};

    for (const Move& move : eatenMoves)
    {
        const std::map<PIECEID, int> weightPairs{
            { R_KING, 5 },
            { R_ROOK, 4 },
            { R_CANNON, 3 },
            { R_KNIGHT, 3 },
            { R_BISHOP, 2 },
            { R_GUARD, 2 },
            { R_PAWN, 1 },
        };
        PIECEID attacker = abs(board.pieceidOn(move.x2, move.y2));
        PIECEID captured = abs(board.pieceidOn(move.x2, move.y2));
        int moveWeight = 10 * (8 - weightPairs.at(attacker)) + weightPairs.at(captured);
        moveWeights.emplace_back(moveWeight);
        orderMap[moveWeight].emplace_back(move);
    }

    std::sort(moveWeights.begin(), moveWeights.end(), std::less<int>());
    moveWeights.erase(std::unique(moveWeights.begin(), moveWeights.end()), moveWeights.end());

    for (int weight : moveWeights)
    {
        result.insert(result.end(), orderMap[weight].begin(), orderMap[weight].end());
    }
    for (const Move& move : moves)
    {
        if (move.x1 != -1)
            result.emplace_back(move);
    }
    moves = result;
}
