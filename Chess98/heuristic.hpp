#include "base.hpp"

/* ***** 历史启发 ***** */

/// @brief 历史表
std::array<std::array<int, 90>, 90> historyTable{};

/// @brief 历史启发
class HistoryHeuristic
{
public:
    static void init();
    static void sort(MOVES& moves);
    static void add(Move move, int depth);
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
    for (std::array<int, 90> &i : historyTable)
    {
        for (int& j : i)
        {
            j = 0;
        }
    }
}

/// @brief 历史表排序
/// @param moves
/// @return
void HistoryHeuristic::sort(MOVES& moves)
{
    MOVES result{};
    std::vector<int> valList{};
    std::map<int, std::vector<Move>> valMoves{};
    for (const Move& move : moves)
    {
        int pos1 = toIndex(move.x1, move.y1);
        int pos2 = toIndex(move.x2, move.y2);
        int val = historyTable[pos1][pos2];
        valMoves[val].emplace_back(move);
        valList.emplace_back(val);
    }
    std::sort(valList.begin(), valList.end(), std::greater<int>());
    int m = int(std::unique(valList.begin(), valList.end()) - valList.begin());
    for (int i = 0; i < m; i++)
    {
        for (Move move : valMoves[valList[i]])
        {
            result.emplace_back(move);
        }
    }
    moves = result;
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

/* ***** 杀手启发 ***** */

std::map<int, std::array<Move, 2>> killerMoves{};

/// @brief 杀手启发
class KillerHeuristic
{
public:
    static void sort(int depth, MOVES& moves);
    static void add(int depth, Move move);
};

/// @brief 杀手启发排序
/// @param moves
void KillerHeuristic::sort(int depth, MOVES& moves)
{
    for (Move& move : moves)
    {
        if (killerMoves[depth][0] == move)
        {
            std::swap(move, moves[0]);
        }
        if (killerMoves[depth][1] == move)
        {
            std::swap(move, moves[1]);
        }
    }
}

/// @brief 在杀手表中增加值
/// @param moves
void KillerHeuristic::add(int depth, Move cutMove)
{
    // 若不存在key则增加一个key
    if (killerMoves.count(depth) == 0)
    {
        killerMoves[depth] = {};
    }
    if (cutMove != killerMoves[depth][0])
    {
        killerMoves[depth][1] = killerMoves[depth][0];
        killerMoves[depth][0] = cutMove;
    }
}
