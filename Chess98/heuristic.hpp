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
    std::array<std::array<std::array<int, 90>, 90>,2> historyTable{};

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
            int teamID = (move.attacker.team() + 1) >> 1;
            assert(teamID >= 0 && teamID <= 1);
            move.moveType = history;
            move.val = historyTable[teamID][pos1][pos2];
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
    int teamID = (move.attacker.team() + 1) >> 1;
    assert(teamID >= 0 && teamID <= 1);
    this->historyTable.at(teamID).at(pos1).at(pos2) += depth * depth;
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
    void add(Board &board, Move &goodMove, int vl, int type, int depth);
    Move get(Board &board,int& vl, int vlApha, int vlBeta,int depth);
    int vlAdjust(int vl, int nDistance);

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

int TransportationTable::vlAdjust(int vl, int nDistance)
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

void TransportationTable::add(Board &board, Move &goodMove,int vl,int type, int depth)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList.at(pos);
    if (t.hashLock == 0 || depth >= t.depth) 
    {
        t.hashLock = board.hashLock;
        t.depth = depth;
        if (goodMove.x1 != -1)
        {
            t.goodMove = goodMove;
        }
        t.vl = vl;
        t.type = type;
    }
}

Move TransportationTable::get(Board &board, int& vl, int vlApha, int vlBeta,int depth)
{
    const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
    tItem &t = this->pList.at(pos);
    if (t.hashLock == board.hashLock)
    {
        if (t.depth >= depth)
        {
            if (t.type == exactType)
            {
                vl = this->vlAdjust(t.vl, board.distance);
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
        
        if (isValidMoveInSituation(board, t.goodMove))
        {
            return t.goodMove;
        }
    }
    return Move{};
}
