#pragma once
#include "base.hpp"
#include "hash.hpp"
#include "board.hpp"

// 历史启发
class HistoryHeuristic
{
public:
	HistoryHeuristic() = default;
	void sort(MOVES& moves) const;
	void add(Move move, int depth);
	void reset();

private:
	std::array<std::array<std::array<int, 90>, 90>, 2> historyTable{};

	int toIndex(int x, int y) const
	{
		return 10 * x + y;
	}
};

void HistoryHeuristic::sort(MOVES& moves) const
{
	for (Move& move : moves)
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
		[](Move& first, Move& second) -> bool
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
	std::memset(this->historyTable.data(), 0, sizeof(this->historyTable));
}

// 杀手启发
class KillerTable
{
public:
	KillerTable() = default;
	void reset();
	void add(Board& board, Move move);
	MOVES get(Board& board);

private:
	std::array<std::array<Move, 2>, 128> killerMoves{};
};

void KillerTable::reset()
{
	this->killerMoves.fill({});
}

MOVES KillerTable::get(Board& board)
{
	MOVES results{};
	for (const Move& move : this->killerMoves[board.distance])
	{
		if (isValidMoveInSituation(board, move))
		{
			results.emplace_back(move);
		}
	}
	return results;
}

void KillerTable::add(Board& board, Move move)
{
	std::array<Move, 2>& moves = this->killerMoves[board.distance];
	moves[1] = moves[0];
	moves[0] = move;
}

// 吃子启发
void captureSort(Board& board, MOVES& moves)
{
	const std::map<PIECEID, int> weightPairs{
		{R_KING, 4},
		{R_ROOK, 4},
		{R_CANNON, 3},
		{R_KNIGHT, 3},
		{R_BISHOP, 2},
		{R_GUARD, 2},
		{R_PAWN, 1},
	};
	// 获取吃子着法数量
	int count = 0;
	for (const Move& move : moves)
	{
		if (move.captured.pieceid != EMPTY_PIECEID)
		{
			count++;
		}
	}
	// 提取吃子着法
	std::sort(moves.begin(), moves.end(),
		[&board, &weightPairs](const Move& first, const Move& second) -> bool
		{
			// 把吃子着法放在前面
			if (second.captured.pieceid != EMPTY_PIECEID)
			{
				return false;
			}
		}
	);
	// 对吃子着法进行排序，小子吃大子优先
	std::sort(moves.begin(), moves.begin() + count,
		[&board, &weightPairs](const Move& first, const Move& second) -> bool
		{
			if (first.captured.pieceid == EMPTY_PIECEID || second.captured.pieceid == EMPTY_PIECEID)
			{
				return false;
			}
			int firstWeight = weightPairs.at(std::abs(first.captured.pieceid));
			int secondWeight = weightPairs.at(std::abs(second.captured.pieceid));
			if (firstWeight != secondWeight)
			{
				return firstWeight > secondWeight;
			}
			return first.val > second.val;
		}
	);
}

// 置换表启发
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
	void add(Board& board, Move goodMove, int vl, int type, int depth);
	int getValue(Board& board, int vlApha, int vlBeta, int depth);
	Move getMove(Board& board);
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

void TransportationTable::add(Board& board, Move goodMove, int vl, int type, int depth)
{
	const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
	tItem& t = this->pList.at(pos);
	if (t.hashLock == 0 || depth >= t.depth)
	{
		t.hashLock = board.hashLock;
		t.depth = depth;
		t.goodMove = goodMove;
		t.vl = vl;
		t.type = type;
	}
}

int TransportationTable::getValue(Board& board, int vlApha, int vlBeta, int depth)
{
	const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
	tItem& t = this->pList.at(pos);
	if (t.hashLock == board.hashLock)
	{
		if (t.depth >= depth)
		{
			if (t.type == exactType)
			{
				return this->vlAdjust(t.vl, board.distance);
			}
			else if (t.type == alphaType && t.vl <= vlApha)
			{
				return vlApha;
			}
			else if (t.type == betaType && t.vl >= vlBeta)
			{
				return vlBeta;
			}
		}
	}
	return -INF;
}

Move TransportationTable::getMove(Board& board)
{
	const int pos = static_cast<uint32_t>(board.hashKey) & static_cast<uint32_t>(this->hashMask);
	tItem& t = this->pList.at(pos);
	if (t.hashLock == board.hashLock)
	{
		if (isValidMoveInSituation(board, t.goodMove))
		{
			return t.goodMove;
		}
	}
	return Move{};
}
