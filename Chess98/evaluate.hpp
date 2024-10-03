#pragma once

#include "board.hpp"

const int SCORE_KING = 100000;
const int SCORE_GUARD = 100;
const int SCORE_BISHOP = 100;
const int SCORE_KNIGHT = 500;
const int SCORE_ROOK = 1000;
const int SCORE_CANNON = 500;
const int SCORE_PAWN = 50;

/// @brief 评估算法
class Evaluate
{
public:
    static int evaluate(Board board);
};

/// @brief 评估函数
/// @version 1.0.0 最简单的基本评估
/// @param board
/// @return
int Evaluate::evaluate(Board board)
{
    int redScore = 0;
    int blackScore = 0;

    for (const Piece& piece : board.getAllPieces())
    {
        switch (piece.pieceid)
        {
        case R_KING:
            redScore += SCORE_KING;
            break;
        case R_GUARD:
            redScore += SCORE_GUARD;
            break;
        case R_BISHOP:
            redScore += SCORE_BISHOP;
            break;
        case R_KNIGHT:
            redScore += SCORE_KNIGHT;
            break;
        case R_ROOK:
            redScore += SCORE_ROOK;
            break;
        case R_CANNON:
            redScore += SCORE_CANNON;
            break;
        case R_PAWN:
            redScore += SCORE_PAWN;
            break;

        case B_KING:
            blackScore -= SCORE_KING;
            break;
        case B_GUARD:
            blackScore -= SCORE_GUARD;
            break;
        case B_BISHOP:
            blackScore -= SCORE_BISHOP;
            break;
        case B_KNIGHT:
            blackScore -= SCORE_KNIGHT;
            break;
        case B_ROOK:
            blackScore -= SCORE_ROOK;
            break;
        case B_CANNON:
            blackScore -= SCORE_CANNON;
            break;
        case B_PAWN:
            blackScore -= SCORE_PAWN;
            break;

        default:
            break;
        }
    }
    return redScore + blackScore;
}
