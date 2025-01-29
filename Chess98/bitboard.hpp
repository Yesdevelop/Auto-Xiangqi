#pragma once
#include "base.hpp"

using ROW = std::array<int, 9>;
using COL = std::array<int, 10>;

/// @brief 位棋盘
class BitBoard
{
public:
    BitBoard(PIECEID_MAP pieceidMap)
    {
        for (int x = 0; x < 9; x++)
        {
            for (int y = 0; y < 10; y++)
            {
                if (pieceidMap[x][y] > 0)
                {

                }
            }
        }
    }
    std::array<int, 9> redPiecesRow{};
    std::array<int, 10> redPiecesCol{};

    std::array<int, 9> redRookRow{};
    std::array<int, 10> redRookCol{};

    std::array<int, 9> redKnightRow{};
    std::array<int, 10> redKnightCol{};

    std::array<int, 9> redBishopRow{};
    std::array<int, 10> redBishopCol{};

    std::array<int, 9> redGuardRow{};
    std::array<int, 10> redGuardCol{};

    std::array<int, 9> redKingRow{};
    std::array<int, 10> redKingCol{};

    std::array<int, 9> redCannonRow{};
    std::array<int, 10> redCannonCol{};

    std::array<int, 9> redPawnRow{};
    std::array<int, 10> redPawnCol{};

    std::array<int, 9> blackPiecesRow{};
    std::array<int, 10> blackPiecesCol{};

    std::array<int, 9> blackRookRow{};
    std::array<int, 10> blackRookCol{};

    std::array<int, 9> blackKnightRow{};
    std::array<int, 10> blackKnightCol{};

    std::array<int, 9> blackBishopRow{};
    std::array<int, 10> blackBishopCol{};

    std::array<int, 9> blackGuardRow{};
    std::array<int, 10> blackGuardCol{};

    std::array<int, 9> blackKingRow{};
    std::array<int, 10> blackKingCol{};

    std::array<int, 9> blackCannonRow{};
    std::array<int, 10> blackCannonCol{};

    std::array<int, 9> blackPawnRow{};
    std::array<int, 10> blackPawnCol{};
    
private:
    void setBitPieceRow(ROW& array, int x, int y)
    {
        array[x] |= (1 << y);
    }

    void deleteBitPieceRow(ROW& array, int x, int y)
    {
        array[x] &= ~(1 << y);
    }

    void setBitPieceCol(COL& array, int x, int y)
    {
        array[y] |= (1 << x);
    }

    void deleteBitPieceCol(COL& array, int x, int y)
    {
        array[y] &= ~(1 << x);
    }
};
