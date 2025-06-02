#pragma once
#include "base.hpp"
#include "hash.hpp"
#include "board.hpp"

// History Heuristic
class HistoryHeuristic
{
public:
    HistoryHeuristic() = default;
    void sort(MOVES &moves) const;
    void add(Move move, int depth);
    void reset();

private:
    std::array<std::array<int, 90>, 90> historyTable{};

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
    // vl history compare
    std::sort(moves.begin(), moves.end(),
              [](Move &first, Move &second) -> bool
              {
                  if (first.moveType != second.moveType)
                  {
                      return first.moveType > second.moveType;
                  }
                  return first.val > second.val;
              });
}

void HistoryHeuristic::add(Move move, int depth)
{
    int pos1 = this->toIndex(move.x1, move.y1);
    int pos2 = this->toIndex(move.x2, move.y2);
    this->historyTable.at(pos1).at(pos2) += depth * depth;
}

void HistoryHeuristic::reset()
{
    this->historyTable.fill({});
}

// Killer Table
class KillerTable
{
public:
    KillerTable() = default;
    void reset();
    void add(Board &board, Move move);
    MOVES get(Board &board);

private:
    std::array<std::array<Move, 2>, 128> killerMoves{};
};

void KillerTable::reset()
{
    this->killerMoves.fill({});
}

MOVES KillerTable::get(Board &board)
{
    MOVES results{};
    for (const Move &move : this->killerMoves[board.distance])
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
    std::array<Move, 2> &moves = this->killerMoves[board.distance];
    moves[1] = moves[0];
    moves[0] = move;
}

// Transportation Table
class TransportationTable
{
public:
    TransportationTable(int hashLevel = 24)
    {
        this->hashSize = (1 << hashLevel);
        this->hashMask = this->hashSize - 1;
        this->pList.resize(this->hashSize);
    }

    void reset();
    void add(Board &board, Move &goodMove);
    Move get(Board &board);

private:
    std::vector<tItem> pList{};
    int hashMask = 0;
    int hashSize = 0;
};

void TransportationTable::reset()
{
    this->pList = std::vector<tItem>{};
    this->pList.resize(this->hashSize);
}

void TransportationTable::add(Board &board, Move &goodMove)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList.at(pos);
    t.hashLock = board.hashLock;
    t.goodMove = goodMove;
}

Move TransportationTable::get(Board &board)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList.at(pos);
    if (t.hashLock == 0 || t.hashLock == board.hashLock)
    {
        if (isValidMoveInSituation(board, t.goodMove))
        {
            return t.goodMove;
        }
    }
    return Move{};
}
