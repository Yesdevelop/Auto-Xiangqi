#pragma once
#include "base.hpp"
#include "board.hpp"

enum moveType {
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
	void sort(MOVES& moves);
	void add(Move move, int depth);

	static bool vlHisCompare(Move& first, Move& second)
	{
		if (first.moveType != second.moveType) {
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
void HistoryHeuristic::sort(MOVES& moves)
{
	for (Move& move : moves) {
		if (move.moveType <= killer) {
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
void captureHeuristic(Board& board, MOVES& moves)
{
	const std::map<PIECEID, int> weightPairs {
		{ R_KING, 31 },
		{ R_ROOK, 15 },
		{ R_CANNON, 7 },
		{ R_KNIGHT, 7 },
		{ R_BISHOP, 3 },
		{ R_GUARD, 3 },
		{ R_PAWN, 1 },
	};

	for (Move& move : moves) {
		if (move.moveType <= capture) {
			if (board.pieceidOn(move.x2, move.y2) != 0) {
				PIECEID attacker = abs(board.pieceidOn(move.x2, move.y2));
				PIECEID captured = abs(board.pieceidOn(move.x2, move.y2));
				move.moveType = capture;
				move.val = weightPairs.at(captured) / weightPairs.at(attacker);
			}
		}
	}
}
