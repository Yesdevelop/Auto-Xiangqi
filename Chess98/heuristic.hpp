#pragma once
#include "base.hpp"
#include "hash.hpp"
#include "board.hpp"

// 历史启发
enum moveType
{
    normal = 0,
    capture = 1,
    history = 2,
    killer = 3,
    hash = 4
};

class HistoryHeuristic
{
public:
    void init() const;
    void sort(MOVES &moves) const;
    void add(Move move, int depth);

    static bool vlHisCompare(Move &first, Move &second)
    {
        if (first.moveType != second.moveType)
        {
            return first.moveType > second.moveType;
        }
        return first.val > second.val;
    }

    std::array<std::array<int, 90>, 90> historyTable{};
};

/// @brief 二维坐标转索引
/// @param x
/// @param y
/// @return
inline int toIndex(int x, int y)
{
    return 10 * x + y;
}

/// @brief 初始化
void HistoryHeuristic::init() const
{
}

/// @brief 历史表排序
/// @param moves
/// @return
void HistoryHeuristic::sort(MOVES &moves) const
{
    for (Move &move : moves)
    {
        if (move.moveType <= history)
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
    try {
        this->historyTable.at(pos1).at(pos2) += depth * depth;
    }
    catch (std::exception) {
        std::cout << move.id << std::endl;
        std::cout << pos1 << std::endl << pos2 << std::endl;
        while (true);
    }
}

class KillerTable
{
public:
    ~KillerTable();
    void init();
    bool initDone();
    void reset();
    void add(Board &board, Move move);
    std::vector<Move> get(Board &board);

private:
    const int maxKillerDistance = 128;
    const int width = 2;
    std::vector<MOVES> KillerMoveList;
};

KillerTable::~KillerTable()
{
    for (auto &MoveVec : this->KillerMoveList)
    {
        std::vector<Move>().swap(MoveVec);
    }
    std::vector<MOVES>().swap(this->KillerMoveList);
}

void KillerTable::init()
{
    KillerMoveList.resize(this->maxKillerDistance);
    for (auto &moveVec : this->KillerMoveList)
    {
        moveVec = MOVES(this->width, Move());
    }
}

bool KillerTable::initDone()
{
    return !(this->KillerMoveList.empty());
}

void KillerTable::reset()
{
    for (auto &moveVec : this->KillerMoveList)
    {
        moveVec = MOVES(this->width, Move());
    }
}

MOVES KillerTable::get(Board &board)
{
    assert(board.distance < this->maxKillerDistance);
    MOVES results{};
    for (auto &move : this->KillerMoveList[board.distance])
    {
        if (isValidMoveInSituation(board, move))
        {
            results.emplace_back(move);
        }
    }
    return results;
}

void KillerTable::add(Board &board, Move move)
{
    assert(board.distance < this->maxKillerDistance);
    MOVES &moveVec = this->KillerMoveList[board.distance];
    moveVec[1] = moveVec[0];
    moveVec[0] = move;
}

// 置换表启发
enum nodeType
{
    noneType = 0,
    alphaType = 1,
    betaType = 2,
    exactType = 3,
};

struct tItem
{
    Move goodMove;
    int32 hashLock = 0;
};

class TransportationTable
{
public:
    ~TransportationTable();
    void init(int hashLevel = 16);
    bool initDone();
    void reset();
    void add(Board &board, Move &goodMove);
    void get(Board &board, Move &goodMove);

private:
    std::vector<tItem> *pList = nullptr;
    int hashMask = 0;
    int hashSize = 0;
};

void TransportationTable::init(int hashLevel)
{
    if (this->pList != nullptr)
    {
        delete this->pList;
    }
    this->pList = new std::vector<tItem>;
    this->hashSize = (1 << hashLevel);
    this->hashMask = this->hashSize - 1;
    this->pList->resize(this->hashSize);
}

bool TransportationTable::initDone()
{
    return (this->pList != nullptr);
}

void TransportationTable::reset()
{
    if (this->pList != nullptr)
    {
        std::vector<tItem>().swap(*this->pList);
        this->pList->resize(this->hashSize);
    }
}

TransportationTable::~TransportationTable()
{
    if (pList != nullptr)
    {
        delete pList;
        pList = nullptr;
    }
}

void TransportationTable::add(Board &board, Move &goodMove)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList->at(pos);
    t.hashLock = board.hashLock;
    t.goodMove = goodMove;
}

void TransportationTable::get(Board &board, Move &goodMove)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList->at(pos);
    if (t.hashLock == 0 || t.hashLock == board.hashLock)
    {
        if (isValidMoveInSituation(board, t.goodMove))
        {
            goodMove = t.goodMove;
        }
    }
}
