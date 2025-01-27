#pragma once
#include "base.hpp"

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
