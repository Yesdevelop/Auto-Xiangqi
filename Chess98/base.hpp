/**
 * 说明见 /DEV.md
 */

#pragma once
#ifdef _WIN32
#include <windows.h>
#endif
#include <cassert>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <string>
#include <vector>
#include <cstring>
#include <array>
#include <map>
#include <cmath>
#include <thread>

const int INF = 1000000;
const int BAN = INF - 2000;
const int ILLEGAL_VAL = INF * 2;
using U64 = unsigned long long;

using PIECE_INDEX = int;
const PIECE_INDEX EMPTY_INDEX = -1;

using PIECEID = int;
const PIECEID EMPTY_PIECEID = 0;
const PIECEID R_KING = 1;
const PIECEID R_GUARD = 2;
const PIECEID R_BISHOP = 3;
const PIECEID R_KNIGHT = 4;
const PIECEID R_ROOK = 5;
const PIECEID R_CANNON = 6;
const PIECEID R_PAWN = 7;
const PIECEID B_KING = -1;
const PIECEID B_GUARD = -2;
const PIECEID B_BISHOP = -3;
const PIECEID B_KNIGHT = -4;
const PIECEID B_ROOK = -5;
const PIECEID B_CANNON = -6;
const PIECEID B_PAWN = -7;
const PIECEID OVERFLOW_PIECEID = 8;

using TEAM = int;
const TEAM EMPTY_TEAM = 0;
const TEAM RED = 1;
const TEAM BLACK = -1;
const TEAM OVERFLOW_TEAM = 2;

using PIECEID_MAP = std::array<std::array<PIECEID, 10>, 9>;
PIECEID_MAP DEFAULT_MAP{
    {{R_ROOK, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_ROOK},
     {R_KNIGHT, 0, R_CANNON, 0, 0, 0, 0, B_CANNON, 0, B_KNIGHT},
     {R_BISHOP, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_BISHOP},
     {R_GUARD, 0, 0, 0, 0, 0, 0, 0, 0, B_GUARD},
     {R_KING, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_KING},
     {R_GUARD, 0, 0, 0, 0, 0, 0, 0, 0, B_GUARD},
     {R_BISHOP, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_BISHOP},
     {R_KNIGHT, 0, R_CANNON, 0, 0, 0, 0, B_CANNON, 0, B_KNIGHT},
     {R_ROOK, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_ROOK}}};

std::map<PIECEID, std::string> PIECE_NAME_PAIRS {
    {R_KING, "RK"},
    {R_GUARD, "RG"},
    {R_BISHOP, "RB"},
    {R_KNIGHT, "RN"},
    {R_ROOK, "RR"},
    {R_CANNON, "RC"},
    {R_PAWN, "RP"},
    {B_KING, "BK"},
    {B_GUARD, "BG"},
    {B_BISHOP, "BB"},
    {B_KNIGHT, "BN"},
    {B_ROOK, "BR"},
    {B_CANNON, "BC"},
    {B_PAWN, "BP"},
    {EMPTY_PIECEID, "__"},
    {OVERFLOW_PIECEID, "  "}
};

std::map<std::string, PIECEID> NAME_PIECE_PAIRS {
    {"RK", R_KING},
    {"RG", R_GUARD},
    {"RB", R_BISHOP},
    {"RN", R_KNIGHT},
    {"RR", R_ROOK},
    {"RC", R_CANNON},
    {"RP", R_PAWN},
    {"BK", B_KING},
    {"BG", B_GUARD},
    {"BB", B_BISHOP},
    {"BN", B_KNIGHT},
    {"BR", B_ROOK},
    {"BC", B_CANNON},
    {"BP", B_PAWN},
    {"__", EMPTY_PIECEID},
    {"  ", OVERFLOW_PIECEID}
};

/// @brief 棋子类
class Piece
{
public:
    Piece() = default;
    Piece(PIECEID pieceid, int x, int y, PIECE_INDEX pieceIndex)
        : pieceid(pieceid),
          x(x),
          y(y),
          pieceIndex(pieceIndex) {}

    TEAM team() const
    {
        if (this->pieceid == EMPTY_PIECEID)
        {
            return EMPTY_TEAM;
        }
        else if (this->pieceid == OVERFLOW_PIECEID)
        {
            return OVERFLOW_TEAM;
        }
        else if (this->pieceid > 0)
        {
            return RED;
        }
        else
        {
            return BLACK;
        }
    }

    PIECEID pieceid = EMPTY_PIECEID;
    int x = -1;
    int y = -1;
    PIECE_INDEX pieceIndex = -1;
    bool isLive = true;
};

using PIECES = std::vector<Piece>;

/// @brief 着法类
class Move
{
public:
    Move() = default;
    Move(int x1, int y1, int x2, int y2, int val = 0, int moveType = 0)
        : x1(x1),
          y1(y1),
          x2(x2),
          y2(y2),
          id(x1 * 1000 + y1 * 100 + x2 * 10 + y2),
          val(val),
          moveType(moveType) {}
    int id = -1;
    int x1 = -1;
    int y1 = -1;
    int x2 = -1;
    int y2 = -1;
    int val = 0;
    int moveType = 0;
    bool isCheckingMove = false;
    Piece starter{};
    Piece captured{};

    constexpr bool operator==(const Move &move) const
    {
        return this->id == move.id;
    }

    constexpr bool operator!=(const Move &move) const
    {
        return this->id != move.id;
    }
};

using MOVES = std::vector<Move>;

/// @brief 根节点类，记录搜索结果
class Result
{
public:
    Result(Move move, int score) : move(move), val(score) {}
    Move move{};
    int val = 0;
};
