#pragma once
#include "base.hpp"
#include "hash.hpp"
#include "board.hpp"

enum moveType
{
    normal = 0,
    capture = 1,
    history = 2,
    killer = 3,
    hash = 4

};

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
        if (first.moveType != second.moveType)
        {
            return first.moveType > second.moveType;
        }
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
        if (move.moveType <= killer)
        {
            int pos1 = toIndex(move.x1, move.y1);
            int pos2 = toIndex(move.x2, move.y2);
            move.moveType = history;
            move.val = historyTable[pos1][pos2];
        }
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

/* ***** 吃子启发 ***** */
void captureHeuristic(Board &board, MOVES &moves)
{
    const std::map<PIECEID, int> weightPairs{
        {R_KING, 31},
        {R_ROOK, 15},
        {R_CANNON, 7},
        {R_KNIGHT, 7},
        {R_BISHOP, 3},
        {R_GUARD, 3},
        {R_PAWN, 1},
    };

    for (Move &move : moves)
    {
        if (move.moveType <= capture)
        {
            if (board.pieceidOn(move.x2, move.y2) != 0)
            {
                PIECEID attacker = abs(board.pieceidOn(move.x2, move.y2));
                PIECEID captured = abs(board.pieceidOn(move.x2, move.y2));
                move.moveType = capture;
                move.val = weightPairs.at(captured) - weightPairs.at(attacker);
            }
        }
    }
}

enum nodeType
{
    noneType = 0,
    alphaType = 1,
    betaType = 2,
    exactType = 3,
};

/* ***** 置换启发 ***** */

/// @brief 置换启发

struct tItem
{
    nodeType type = noneType;
    int vl = 0;
    int alpha = 0;
    int beta = 0;
    int depth = 0;
    int32 hashLock = 0;
    bool risky = false;
};

class tt
{
public:
    ~tt();
    void init(int hashLevel = 16);
    bool initDone();
    void reset();
    void add(int32 hashKey, int32 hashLock, int vl, nodeType type, int depth, bool risky);
    void get(int32 hashKey, int32 hashLock, int &vl, int vlApha, int vlBeta, int depth, int nDistance);
    int vlAdjust(int vl, int nDistance);

private:
    tItem *pList = nullptr;
    int hashMask = 0;
    int hashSize = 0;
};

void tt::init(int hashLevel)
{
    this->hashSize = (1 << hashLevel);
    this->hashMask = this->hashSize - 1;
    pList = new tItem[this->hashSize];
}

bool tt::initDone()
{
    return (this->pList != nullptr);
}

void tt::reset()
{
    memset(this->pList, 0, sizeof(tItem) * this->hashSize);
}

tt::~tt()
{
    if (pList != nullptr)
    {
        delete[] pList;
        pList = nullptr;
    }
}

int tt::vlAdjust(int vl, int nDistance)
{
    if (std::abs(vl) >= BAN)
    {
        if (vl < 0)
        {
            return vl + nDistance;
        }
        if (vl > 0)
        {
            return vl - nDistance;
        }
    }
    return vl;
}

void tt::add(int32 hashKey, int32 hashLock, int vl, nodeType type, int depth, bool risky)
{
    int pos = hashKey & this->hashMask;
    tItem &t = this->pList[pos];
    if (t.type == noneType)
    {
        t.alpha = vlApha;
        t.beta = vlBeta;
        t.depth = depth;
        t.hashLock = hashLock;
        t.type = type;
        t.risky = risky;
    }
    else if (depth >= t.depth)
    {
        // 避免向前裁剪覆盖正常的搜索结果
        bool riskyCoverage = (!t.risky && risky);
        if (!riskyCoverage)
        {
            t.depth = depth;
            t.hashLock = hashLock;
            t.type = type;
            t.risky = risky;
        }
    }
}

void tt::get(int32 hashKey, int32 hashLock, int &vl, int vlApha, int vlBeta, int depth, int nDistance)
{
    int pos = hashKey & this->hashMask;
    tItem &t = this->pList[pos];
    if (t.type != noneType && t.hashLock == hashLock && t.depth >= depth)
    {
        if (t.type == exactType)
        {
            vl = this->vlAdjust(vl, nDistance);
        }
        else if (t.type == alphaType && t.vl <= vlApha)
        {
            vl = vlApha;
        }
        else if (t.type == betaType && t.vl >= vlBeta)
        {
            vl = vlBeta;
        }
    }
}
