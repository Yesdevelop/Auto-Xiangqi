#pragma once
#include "board.hpp"

/// @brief 判断当前一方是否被将军
/// @param board
/// @return
bool inCheck(Board &board)
{
    Piece *king = board.team == RED ? board.pieceRedKing : board.pieceRedKing;
    // 判断敌方的兵是否在附近
    bool c1 = abs(board.pieceidOn(king->x + 1, king->y)) == R_PAWN;
    bool c2 = abs(board.pieceidOn(king->x - 1, king->y)) == R_PAWN;
    bool c3 = abs(board.pieceidOn(king->x, king->y + 1)) == R_PAWN;
    // 判断敌方的马是否在附近
    auto piece1 = board.pieceidOn(king->x + 2, king->y + 1);
    auto piece2 = board.pieceidOn(king->x - 2, king->y + 1);
    auto piece3 = board.pieceidOn(king->x + 2, king->y - 1);
    auto piece4 = board.pieceidOn(king->x - 2, king->y - 1);
    auto piece5 = board.pieceidOn(king->x + 1, king->y + 2);
    auto piece6 = board.pieceidOn(king->x + 1, king->y - 2);
    auto piece7 = board.pieceidOn(king->x - 1, king->y + 2);
    auto piece8 = board.pieceidOn(king->x - 1, king->y - 2);
    bool c4 = abs(piece1) == R_KNIGHT && (board.team > 0 ? (piece1 < 0) : (piece1 > 0));
    bool c5 = abs(piece2) == R_KNIGHT && (board.team > 0 ? (piece2 < 0) : (piece2 > 0));
    bool c6 = abs(piece3) == R_KNIGHT && (board.team > 0 ? (piece3 < 0) : (piece3 > 0));
    bool c7 = abs(piece4) == R_KNIGHT && (board.team > 0 ? (piece4 < 0) : (piece4 > 0));
    bool c8 = abs(piece5) == R_KNIGHT && (board.team > 0 ? (piece5 < 0) : (piece5 > 0));
    bool c9 = abs(piece6) == R_KNIGHT && (board.team > 0 ? (piece6 < 0) : (piece6 > 0));
    bool c10 = abs(piece7) == R_KNIGHT && (board.team > 0 ? (piece7 < 0) : (piece8 > 0));
    bool c11 = abs(piece8) == R_KNIGHT && (board.team > 0 ? (piece8 < 0) : (piece8 > 0));
    // 判断是否被将军
    bool condition = c1 || c2 || c3 || c4 || c5 || c6 || c7 || c8 || c9 || c10 || c11;
    if (condition == true)
        return true;

    // 白脸将、车、炮
    bool barrierDetected = false;
    for (int x = king->x + 1; x < 9; x++)
    {
        PIECEID pieceid = board.pieceidOn(x, king->y);
        TEAM team = board.teamOn(x, king->y);
        if (abs(pieceid) == R_ROOK &&
            team != board.team &&
            barrierDetected == false)
        {
            return true;
        }
        else if (abs(pieceid) == R_CANNON &&
                 team != board.team &&
                 barrierDetected == true)
        {
            return true;
        }
        else if (pieceid != 0)
        {
            barrierDetected = true;
        }
    }
    barrierDetected = false;
    for (int x = king->x - 1; x >= 0; x--)
    {
        PIECEID pieceid = board.pieceidOn(x, king->y);
        TEAM team = board.teamOn(x, king->y);
        if (abs(pieceid) == R_ROOK &&
            team != board.team &&
            barrierDetected == false)
        {
            return true;
        }
        else if (abs(pieceid) == R_CANNON &&
                 team != board.team &&
                 barrierDetected == true)
        {
            return true;
        }
        else if (pieceid != 0)
        {
            barrierDetected = true;
        }
    }
    barrierDetected = false;
    if (board.team == RED)
    {
        for (int y = king->y + 1; y < 10; y++)
        {
            PIECEID pieceid = board.pieceidOn(king->x, y);
            TEAM team = board.teamOn(king->x, y);
            if ((abs(pieceid) == R_ROOK ||
                 abs(pieceid) == R_KING) &&
                team != board.team &&
                barrierDetected == false)
            {
                return true;
            }
            else if (abs(pieceid) == R_CANNON &&
                     team != board.team &&
                     barrierDetected == true)
            {
                return true;
            }
            else if (pieceid != 0)
            {
                barrierDetected = true;
            }
        }
    }
    else
    {
        for (int y = king->y - 1; y >= 0; y--)
        {
            PIECEID pieceid = board.pieceidOn(king->x, y);
            TEAM team = board.teamOn(king->x, y);
            if ((abs(pieceid) == R_ROOK ||
                 abs(pieceid) == R_KING) &&
                team != board.team &&
                barrierDetected == false)
            {
                return true;
            }
            else if (abs(pieceid) == R_CANNON &&
                     team != board.team &&
                     barrierDetected == true)
            {
                return true;
            }
            else if (pieceid != 0)
            {
                barrierDetected = true;
            }
        }
    }
    return false;
}
