#pragma once
#include "base.hpp"
#include "hash.hpp"
#include "board.hpp"

// History Heuristic
class HistoryHeuristic
{
public:
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

private:
    int toIndex(int x, int y) const
    {
        return 10 * x + y;
    }
};

void HistoryHeuristic::sort(MOVES &moves) const
{
    for (Move &move : moves)
    {
        if (move.moveType <= history)
        {
            int pos1 = this->toIndex(move.x1, move.y1);
            int pos2 = this->toIndex(move.x2, move.y2);
            move.moveType = history;
            move.val = historyTable[pos1][pos2];
        }
    }
    std::sort(moves.begin(), moves.end(), vlHisCompare);
}

void HistoryHeuristic::add(Move move, int depth)
{
    int pos1 = this->toIndex(move.x1, move.y1);
    int pos2 = this->toIndex(move.x2, move.y2);
    try {
        this->historyTable.at(pos1).at(pos2) += depth * depth;
    }
    catch (std::exception) {
        std::cout << move.id << std::endl;
        std::cout << pos1 << std::endl << pos2 << std::endl;
        while (true);
    }
}


// Killer Heuristic
class KillerTable
{
public:
    void init();
    bool initDone();
    void reset();
    void add(Board &board, Move move);
    MOVES get(Board &board);

private:
    const int maxKillerDistance = 128;
    const int width = 2;
    std::vector<MOVES> killerMoves{};
};

void KillerTable::init()
{
    killerMoves.resize(this->maxKillerDistance);
    for (auto &moveVec : this->killerMoves)
    {
        moveVec = MOVES(this->width, Move());
    }
}

bool KillerTable::initDone()
{
    return !(this->killerMoves.empty());
}

void KillerTable::reset()
{
    for (auto &moveVec : this->killerMoves)
    {
        moveVec = MOVES(this->width, Move());
    }
}

MOVES KillerTable::get(Board &board)
{
    assert(board.distance < this->maxKillerDistance);
    MOVES results{};
    for (auto &move : this->killerMoves[board.distance])
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
    MOVES &moveVec = this->killerMoves[board.distance];
    moveVec[1] = moveVec[0];
    moveVec[0] = move;
}

// Transportation Table
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
