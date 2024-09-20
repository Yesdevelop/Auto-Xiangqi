#pragma once

#include "base.hpp"

/// @brief 棋盘类
class Board
{
public:
    Board(PIECEID_MAP pieceidMap);

    Piece findPieceByIndex(PIECE_INDEX pieceIndex);
    Piece findPieceByPosition(int x, int y);

    PIECEID pieceidOn(int x, int y);
    TEAM teamOn(int x, int y);

    std::vector<Piece> getAllPieces();
    std::vector<Piece> getPiecesByTeam(TEAM team);

    Piece doMove(int x1, int y1, int x2, int y2);
    Piece doMove(Move move);
    void undoMove(int x1, int x2, int y1, int y2, Piece eaten);
    void undoMove(Move move, Piece eaten);

    void print();

private:
    PIECEID_MAP pieceidMap{};
    std::array<std::array<int, 10>, 9> pieceIndexMap{};
    std::vector<Piece> pieces{};

    std::vector<PIECE_INDEX> redPieces{};
    std::vector<PIECE_INDEX> blackPieces{};
};

/// @brief 初始化棋盘
/// @param pieceidMap 棋子id位置表，一般传DEFAULT_PIECEID_MAP
Board::Board(PIECEID_MAP pieceidMap)
{
    this->pieceidMap = pieceidMap;
    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 10; y++)
        {
            PIECEID pieceid = this->pieceidMap[x][y];
            if (pieceid != 0)
            {
                Piece piece{ this->pieceidMap[x][y], x, y, (int)this->pieces.size() };
                this->pieces.emplace_back(piece);
                this->pieceIndexMap[x][y] = this->pieces.size() - 1;
                if (pieceid > 0)
                {
                    this->redPieces.emplace_back(this->pieces.size() - 1);
                }
                else
                {
                    this->blackPieces.emplace_back(this->pieces.size() - 1);
                }
            }
            else
            {
                this->pieceIndexMap[x][y] = -1;
            }
        }
    }
}

/// @brief 通过索引号查找piece
/// @param pieceIndex
/// @return
Piece Board::findPieceByIndex(PIECE_INDEX pieceIndex)
{
    if (pieceIndex != -1)
    {
        return this->pieces[pieceIndex];
    }
    else
    {
        std::clog << "try to access an invaild pieceIndex!" << std::endl;
        throw "try to access an invaild pieceIndex";
    }
}

/// @brief 通过位置查找piece（若为空则返回一个index为-1的piece）
/// @param x
/// @param y
/// @return
Piece Board::findPieceByPosition(int x, int y)
{
    if (x >= 0 && x <= 8 && y >= 0 && y <= 9)
    {
        PIECEID pieceid = this->pieceidMap[x][y];
        if (pieceid != 0)
        {
            PIECE_INDEX pieceIndex = this->pieceIndexMap[x][y];
            return this->findPieceByIndex(pieceIndex);
        }
        else
        {
            return Piece{ EMPTY_PIECEID, x, y, -1 };
        }
    }
    else
    {
        return Piece{ OVERFLOW_PIECEID, x, y, -1 };
    }
}

/// @brief 获取指定位置上的pieceid
/// @param x
/// @param y
/// @return
PIECEID Board::pieceidOn(int x, int y)
{
    if (x >= 0 && x <= 8 && y >= 0 && y <= 9)
    {
        return this->pieceidMap[x][y];
    }
    else
    {
        return OVERFLOW_PIECEID;
    }
}

/// @brief 获取指定位置上的队伍
/// @param x
/// @param y
/// @return
TEAM Board::teamOn(int x, int y)
{
    if (x >= 0 && x <= 8 && y >= 0 && y <= 9)
    {
        PIECEID pieceid = this->pieceidMap[x][y];
        if (pieceid > 0)
        {
            return RED;
        }
        else if (pieceid < 0)
        {
            return BLACK;
        }
        else
        {
            return EMPTY_TEAM;
        }
    }
    else
    {
        return OVERFLOW_TEAM;
    }
}

/// @brief 获取棋盘上所有存活的棋子
/// @return
std::vector<Piece> Board::getAllPieces()
{
    std::vector<Piece> result{};
    result.reserve(32);
    for (Piece piece : this->pieces)
    {
        if (piece.isLive == true)
        {
            result.emplace_back(piece);
        }
    }
    return result;
}

/// @brief 获取指定队伍的所有存活的棋子
/// @param team
/// @return
std::vector<Piece> Board::getPiecesByTeam(TEAM team)
{
    std::vector<Piece> result{};
    std::vector<Piece> allPieces = this->getAllPieces();
    for (Piece piece : allPieces)
    {
        if (piece.getTeam() == team)
        {
            result.emplace_back(piece);
        }
    }
    return result;
}

/// @brief 步进
/// @param x1
/// @param y1
/// @param x2
/// @param y2
/// @return 被吃掉的子
Piece Board::doMove(int x1, int y1, int x2, int y2)
{
    Piece eaten = this->findPieceByPosition(x2, y2);
    Piece attackStarter = this->findPieceByPosition(x1, y1);

    // 维护棋盘的棋子追踪
    this->pieceidMap[x2][y2] = this->pieceidMap[x1][y1];
    this->pieceidMap[x1][y1] = 0;
    this->pieceIndexMap[x2][y2] = this->pieceIndexMap[x1][y1];
    this->pieceIndexMap[x1][y1] = -1;
    this->pieces[attackStarter.pieceIndex].x = x2;
    this->pieces[attackStarter.pieceIndex].y = y2;
    if (eaten.pieceIndex != -1)
    {
        this->pieces[eaten.pieceIndex].isLive = false;
    }

    return eaten;
}

/// @brief 步进
/// @param move
/// @return 被吃掉的子
Piece Board::doMove(Move move)
{
    return this->doMove(move.x1, move.y1, move.x2, move.y2);
}

/// @brief 撤销步进
/// @param x1
/// @param y1
/// @param x2
/// @param y2
/// @param eaten
void Board::undoMove(int x1, int y1, int x2, int y2, Piece eaten)
{
    Piece attackStarter = this->findPieceByPosition(x2, y2);

    // 维护棋盘的棋子追踪
    this->pieceidMap[x1][y1] = this->pieceidMap[x2][y2];
    this->pieceidMap[x2][y2] = eaten.pieceid;
    this->pieceIndexMap[x1][y1] = this->pieceIndexMap[x2][y2];
    this->pieceIndexMap[x2][y2] = eaten.pieceIndex;
    this->pieces[attackStarter.pieceIndex].x = x1;
    this->pieces[attackStarter.pieceIndex].y = y1;
    if (eaten.pieceIndex != -1)
    {
        this->pieces[eaten.pieceIndex].isLive = true;
    }
}

/// @brief 撤销步进
/// @param move
/// @param eaten
void Board::undoMove(Move move, Piece eaten)
{
    this->undoMove(move.x1, move.y1, move.x2, move.y2, eaten);
}

void Board::print()
{
    for (int i = -1; i <= 8; i++)
    {
        for (int j = -1; j <= 9; j++)
        {
            if (i == -1)
            {
                if (j == -1)
                {
                    std::cout << "X ";
                }
                else
                {
                    std::cout << j << " ";
                }
            }
            else
            {
                if (j == -1)
                {
                    std::cout << i << " ";
                }
                else
                {

                    std::cout << getPieceName(this->pieceidOn(i, j));
                }
            }
        }
        std::cout << std::endl;
    }
}
