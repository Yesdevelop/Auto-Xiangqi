#pragma once
#include "board.hpp"

/// @brief 判断当前一方是否被将军
/// @param board
/// @return
bool inCheck(Board &board)
{
    Piece *king = board.team == RED ? board.pieceRedKing : board.pieceBlackKing;
    int x = king->x;
    int y = king->y;
    int team = king->getTeam();

    // 判断敌方的兵是否在附近
    bool c1 = abs(board.pieceidOn(king->x + 1, king->y)) == R_PAWN;
    bool c2 = abs(board.pieceidOn(king->x - 1, king->y)) == R_PAWN;
    bool c3 = abs(board.pieceidOn(king->x, (king->getTeam() == RED ? king->y - 1 : king->y + 1))) == R_PAWN;
    if (c1 || c2 || c3)
    {
        return true;
    }

    // 判断敌方的马是否在附近
    PIECEID piece1 = 0;
    PIECEID piece2 = 0;
    PIECEID piece3 = 0;
    PIECEID piece4 = 0;
    PIECEID piece5 = 0;
    PIECEID piece6 = 0;
    PIECEID piece7 = 0;
    PIECEID piece8 = 0;
    if (board.pieceidOn(king->x + 1, king->y) != 0)
    {
        piece1 = board.pieceidOn(king->x + 2, king->y + 1);
        piece2 = board.pieceidOn(king->x + 2, king->y - 1);
    }
    if (board.pieceidOn(king->x - 1, king->y) != 0)
    {
        piece3 = board.pieceidOn(king->x - 2, king->y + 1);
        piece4 = board.pieceidOn(king->x - 2, king->y - 1);
    }
    if (board.pieceidOn(king->x, king->y + 1) != 0)
    {
        piece5 = board.pieceidOn(king->x + 1, king->y + 2);
        piece6 = board.pieceidOn(king->x - 1, king->y + 2);
    }
    if (board.pieceidOn(king->x, king->y - 1) != 0)
    {
        piece7 = board.pieceidOn(king->x + 1, king->y - 2);
        piece8 = board.pieceidOn(king->x - 1, king->y - 2);
    }
    bool c4 = abs(piece1) == R_KNIGHT && (board.team > 0 ? (piece1 < 0) : (piece1 > 0));
    bool c5 = abs(piece2) == R_KNIGHT && (board.team > 0 ? (piece2 < 0) : (piece2 > 0));
    bool c6 = abs(piece3) == R_KNIGHT && (board.team > 0 ? (piece3 < 0) : (piece3 > 0));
    bool c7 = abs(piece4) == R_KNIGHT && (board.team > 0 ? (piece4 < 0) : (piece4 > 0));
    bool c8 = abs(piece5) == R_KNIGHT && (board.team > 0 ? (piece5 < 0) : (piece5 > 0));
    bool c9 = abs(piece6) == R_KNIGHT && (board.team > 0 ? (piece6 < 0) : (piece6 > 0));
    bool c10 = abs(piece7) == R_KNIGHT && (board.team > 0 ? (piece7 < 0) : (piece8 > 0));
    bool c11 = abs(piece8) == R_KNIGHT && (board.team > 0 ? (piece8 < 0) : (piece8 > 0));

    // 判断是否被将军
    if (c4 || c5 || c6 || c7 || c8 || c9 || c10 || c11)
    {
        return true;
    }

    // 白脸将、车、炮
    // 横向着法
    BITLINE bitlineY = board.getBitLineY(king->y);
    REGION_CANNON regionY = board.bitboard->getCannonRegion(bitlineY, king->x, 8);
    if ((abs(board.pieceidOn(regionY[1] - 1, y)) == R_ROOK || abs(board.pieceidOn(regionY[1] - 1, y)) == R_KING) &&
        board.teamOn(regionY[1] - 1, y) != king->getTeam())
        return true;
    if ((abs(board.pieceidOn(regionY[2] + 1, y)) == R_ROOK || abs(board.pieceidOn(regionY[2] + 1, y)) == R_KING) &&
        board.teamOn(regionY[2] + 1, y) != king->getTeam())
        return true;
    if (abs(board.pieceidOn(regionY[0], y)) == R_CANNON && board.teamOn(regionY[0], y) != king->getTeam())
        return true;
    if (abs(board.pieceidOn(regionY[3], y)) == R_CANNON && board.teamOn(regionY[3], y) != king->getTeam())
        return true;

    // 纵向着法
    BITLINE bitlineX = board.getBitLineX(x);
    REGION_CANNON regionX = board.bitboard->getCannonRegion(bitlineX, y, 9);
    if ((abs(board.pieceidOn(x, regionX[1] - 1)) == R_ROOK || abs(board.pieceidOn(x, regionX[1] - 1)) == R_KING) &&
        board.teamOn(x, regionX[1] - 1) != team)
        return true;
    if ((abs(board.pieceidOn(x, regionX[2] + 1)) == R_ROOK || abs(board.pieceidOn(x, regionX[2] + 1)) == R_KING) &&
        board.teamOn(x, regionX[2] + 1) != team)
        return true;
    if (abs(board.pieceidOn(x, regionX[0])) == R_CANNON && board.teamOn(x, regionX[0]) != team)
        return true;
    if (abs(board.pieceidOn(x, regionX[3])) == R_CANNON && board.teamOn(x, regionX[3]) != team)
        return true;

    return false;
}

/// @brief 是否被保护
/// @param board
/// @param x
/// @param y
/// @return
bool relationship_hasProtector(Board &board, int x, int y)
{
    TEAM team = -board.teamOn(x, y);

    // 兵、将
    if ((abs(board.pieceidOn(x + 1, y)) == R_PAWN ||
         abs(board.pieceidOn(x + 1, y) == R_KING)) &&
        board.teamOn(x + 1, y) != team)
        return true;
    if ((abs(board.pieceidOn(x - 1, y)) == R_PAWN ||
         abs(board.pieceidOn(x - 1, y) == R_KING)) &&
        board.teamOn(x - 1, y) != team)
        return true;
    if (abs(board.pieceidOn(x, (team == RED) ? y - 1 : y + 1)) == R_PAWN &&
        board.teamOn(x, (team == RED) ? y - 1 : y + 1) != team)
        return true;
    if (abs(board.pieceidOn(x, y + 1)) == R_KING && board.teamOn(x, y + 1) != team)
        return true;
    if (abs(board.pieceidOn(x, y - 1)) == R_KING && board.teamOn(x, y - 1) != team)
        return true;

    // 马
    if (board.pieceidOn(x + 1, y) == 0)
    {
        if (abs(board.pieceidOn(x + 2, y + 1)) == R_KNIGHT &&
            board.teamOn(x + 2, y + 1) != team)
            return true;
        if (abs(board.pieceidOn(x + 2, y - 1)) == R_KNIGHT &&
            board.teamOn(x + 2, y - 1) != team)
            return true;
    }
    if (board.pieceidOn(x - 1, y) == 0)
    {
        if (abs(board.pieceidOn(x - 2, y + 1)) == R_KNIGHT &&
            board.teamOn(x - 2, y + 1) != team)
            return true;
        if (abs(board.pieceidOn(x - 2, y - 1)) == R_KNIGHT &&
            board.teamOn(x - 2, y - 1) != team)
            return true;
    }
    if (board.pieceidOn(x, y + 1) == 0)
    {
        if (abs(board.pieceidOn(x + 1, y + 2)) == R_KNIGHT &&
            board.teamOn(x + 1, y + 2) != team)
            return true;
        if (abs(board.pieceidOn(x - 1, y + 2)) == R_KNIGHT &&
            board.teamOn(x - 1, y + 2) != team)
            return true;
    }
    if (board.pieceidOn(x, y - 1) == 0)
    {
        if (abs(board.pieceidOn(x + 1, y - 2)) == R_KNIGHT &&
            board.teamOn(x + 1, y - 2) != team)
            return true;
        if (abs(board.pieceidOn(x - 1, y - 2)) == R_KNIGHT &&
            board.teamOn(x - 1, y - 2) != team)
            return true;
    }

    // 士、象
    if (board.pieceidOn(x + 1, y + 1) == 0)
    {
        if (abs(board.pieceidOn(x + 2, y + 2)) == R_BISHOP &&
            board.teamOn(x + 2, y + 2) != team)
            return true;
    }
    if (board.pieceidOn(x - 1, y + 1) == 0)
    {
        if (abs(board.pieceidOn(x - 2, y + 2)) == R_BISHOP &&
            board.teamOn(x - 2, y + 2) != team)
            return true;
    }
    if (board.pieceidOn(x + 1, y - 1) == 0)
    {
        if (abs(board.pieceidOn(x + 2, y - 2)) == R_BISHOP &&
            board.teamOn(x + 2, y - 2) != team)
            return true;
    }
    if (board.pieceidOn(x - 1, y - 1) == 0)
    {
        if (abs(board.pieceidOn(x - 2, y - 2)) == R_BISHOP &&
            board.teamOn(x - 2, y - 2) != team)
            return true;
    }

    if (abs(board.pieceidOn(x + 1, y + 1)) == R_GUARD &&
        board.teamOn(x + 1, y + 1) != team)
        return true;
    if (abs(board.pieceidOn(x - 1, y + 1)) == R_GUARD &&
        board.teamOn(x - 1, y + 1) != team)
        return true;
    if (abs(board.pieceidOn(x + 1, y - 1)) == R_GUARD &&
        board.teamOn(x + 1, y - 1) != team)
        return true;
    if (abs(board.pieceidOn(x - 1, y - 1)) == R_GUARD &&
        board.teamOn(x - 1, y - 1) != team)
        return true;

    // 车、炮
    BITLINE bitlineY = board.getBitLineY(y);
    REGION_CANNON regionY = board.bitboard->getCannonRegion(bitlineY, x, 8);
    if (abs(board.pieceidOn(regionY[1] - 1, y)) == R_ROOK && board.teamOn(regionY[1] - 1, y) != team)
        return true;
    if (abs(board.pieceidOn(regionY[2] + 1, y)) == R_ROOK && board.teamOn(regionY[2] + 1, y) != team)
        return true;
    if (abs(board.pieceidOn(regionY[0], y)) == R_CANNON && board.teamOn(regionY[0], y) != team)
        return true;
    if (abs(board.pieceidOn(regionY[3], y)) == R_CANNON && board.teamOn(regionY[3], y) != team)
        return true;

    // 纵向着法
    BITLINE bitlineX = board.getBitLineX(x);
    REGION_CANNON regionX = board.bitboard->getCannonRegion(bitlineX, y, 9);
    if (abs(board.pieceidOn(x, regionX[1] - 1)) == R_ROOK && board.teamOn(x, regionX[1] - 1) != team)
        return true;
    if (abs(board.pieceidOn(x, regionX[2] + 1)) == R_ROOK && board.teamOn(x, regionX[2] + 1) != team)
        return true;
    if (abs(board.pieceidOn(x, regionX[0])) == R_CANNON && board.teamOn(x, regionX[0]) != team)
        return true;
    if (abs(board.pieceidOn(x, regionX[3])) == R_CANNON && board.teamOn(x, regionX[3]) != team)
        return true;

    return false;
}

/// @brief 检查是否是过河卒
/// @param board
/// @param x
/// @param y
/// @return
bool isRiveredPawn(Board &board, int x, int y)
{
    PIECEID pieceid = board.pieceidOn(x, y);
    if (pieceid == R_PAWN && y >= 5 && y <= 9)
    {
        return true;
    }
    if (pieceid == B_PAWN && y >= 0 && y <= 4)
    {
        return true;
    }
    return false;
}
