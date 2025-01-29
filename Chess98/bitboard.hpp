#pragma once
#include "base.hpp"

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

    unsigned xBitBoard[9] = {0};
    unsigned yBitBoard[10] = {0};
    
private:
    void setBitPiece(unsigned array[], int x, int y)
    {
        array[y] |= (1 << x);
    }

    void deleteBitPiece(unsigned array[], int x, int y)
    {
        array[y] &= ~(1 << x);
    }
};
