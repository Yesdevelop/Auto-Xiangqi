#pragma once
#include "board.hpp"
#include "utils.hpp"

/// @brief 着法生成
class Moves
{
public:
    static MOVES king(TEAM team, Board &board, int x, int y);

    static MOVES guard(TEAM team, Board &board, int x, int y);

    static MOVES bishop(TEAM team, Board &board, int x, int y);

    static MOVES knight(TEAM team, Board &board, int x, int y);

    static MOVES rook(TEAM team, Board &board, int x, int y);

    static MOVES cannon(TEAM team, Board &board, int x, int y);

    static MOVES pawn(TEAM team, Board &board, int x, int y);

    static MOVES generateMoves(Board &board, int x, int y);

    static MOVES getMoves(Board &board);

    static MOVES getCaptrueMoves(Board &board);

    static MOVES getGoodCaptures(Board &board);
};

/// @brief 生成将帅的着法
MOVES Moves::king(TEAM team, Board &board, int x, int y)
{
    MOVES result{};
    result.reserve(4);

    // 横坐标应当在3, 5之间，纵坐标的话，红方在0, 2之间，黑方在7, 9之间
    const int left = x - 1;
    const int right = x + 1;
    const int up = y + 1;
    const int down = y - 1;

    if (left >= 3 && board.teamOn(left, y) != team)
        result.emplace_back(Move{x, y, left, y});
    if (right <= 5 && board.teamOn(right, y) != team)
        result.emplace_back(Move{x, y, right, y});
    if (team == RED)
    {
        if (up <= 2 && board.teamOn(x, up) != team)
            result.emplace_back(Move{x, y, x, up});
        if (down >= 0 && board.teamOn(x, down) != team)
            result.emplace_back(Move{x, y, x, down});
    }
    else
    {
        if (up <= 9 && board.teamOn(x, up) != team)
            result.emplace_back(Move{x, y, x, up});
        if (down >= 7 && board.teamOn(x, down) != team)
            result.emplace_back(Move{x, y, x, down});
    }

    return result;
}

/// @brief 生成士的着法
MOVES Moves::guard(TEAM team, Board &board, int x, int y)
{
    MOVES result{};
    result.reserve(4);

    // 横坐标也应在3, 5之间，纵坐标的话，红方在0, 2之间，黑方在7, 9之间
    const int left = x - 1;
    const int right = x + 1;
    const int up = y + 1;
    const int down = y - 1;
    if (left >= 3)
    {
        if (team == RED)
        {
            if (board.teamOn(left, up) != team && up <= 2)
                result.emplace_back(Move{x, y, left, up});
            if (board.teamOn(left, down) != team && down >= 0)
                result.emplace_back(Move{x, y, left, down});
        }
        else
        {
            if (board.teamOn(left, up) != team && up <= 9)
                result.emplace_back(Move{x, y, left, up});
            if (board.teamOn(left, down) != team && down >= 7)
                result.emplace_back(Move{x, y, left, down});
        }
    }
    if (right <= 5)
    {
        if (team == RED)
        {
            if (board.teamOn(right, up) != team && up <= 2)
                result.emplace_back(Move{x, y, right, up});
            if (board.teamOn(right, down) != team && down >= 0)
                result.emplace_back(Move{x, y, right, down});
        }
        else
        {
            if (board.teamOn(right, up) != team && up <= 9)
                result.emplace_back(Move{x, y, right, up});
            if (board.teamOn(right, down) != team && down >= 7)
                result.emplace_back(Move{x, y, right, down});
        }
    }

    return result;
}

/// @brief 生成象的着法
MOVES Moves::bishop(TEAM team, Board &board, int x, int y)
{
    MOVES result{};
    result.reserve(4);

    // 横坐标应在0, 9之间，纵坐标的话，红方在0, 4之间，黑方在5, 9之间
    if (team == RED)
    {
        if (board.teamOn(x - 1, y - 1) == EMPTY_TEAM && board.teamOn(x - 2, y - 2) != team)
            result.emplace_back(Move{x, y, x - 2, y - 2});
        if (board.teamOn(x + 1, y - 1) == EMPTY_TEAM && board.teamOn(x + 2, y - 2) != team)
            result.emplace_back(Move{x, y, x + 2, y - 2});
        if (board.teamOn(x - 1, y + 1) == EMPTY_TEAM && board.teamOn(x - 2, y + 2) != team && y + 1 <= 4)
            result.emplace_back(Move{x, y, x - 2, y + 2});
        if (board.teamOn(x + 1, y + 1) == EMPTY_TEAM && board.teamOn(x + 2, y + 2) != team && y + 1 <= 4)
            result.emplace_back(Move{x, y, x + 2, y + 2});
    }
    else
    {
        if (board.teamOn(x - 1, y - 1) == EMPTY_TEAM && board.teamOn(x - 2, y - 2) != team && y - 1 >= 5)
            result.emplace_back(Move{x, y, x - 2, y - 2});
        if (board.teamOn(x + 1, y - 1) == EMPTY_TEAM && board.teamOn(x + 2, y - 2) != team && y - 1 >= 5)
            result.emplace_back(Move{x, y, x + 2, y - 2});
        if (board.teamOn(x - 1, y + 1) == EMPTY_TEAM && board.teamOn(x - 2, y + 2) != team)
            result.emplace_back(Move{x, y, x - 2, y + 2});
        if (board.teamOn(x + 1, y + 1) == EMPTY_TEAM && board.teamOn(x + 2, y + 2) != team)
            result.emplace_back(Move{x, y, x + 2, y + 2});
    }

    return result;
}

/// @brief 生成马的着法
MOVES Moves::knight(TEAM team, Board &board, int x, int y)
{
    MOVES result{};
    MOVES mayAvailableMoves{};
    if (board.pieceidOn(x, y + 1) == EMPTY_PIECEID)
    {
        mayAvailableMoves.emplace_back(Move{x, y, x + 1, y + 2});
        mayAvailableMoves.emplace_back(Move{x, y, x - 1, y + 2});
    }
    if (board.pieceidOn(x, y - 1) == EMPTY_PIECEID)
    {
        mayAvailableMoves.emplace_back(Move{x, y, x + 1, y - 2});
        mayAvailableMoves.emplace_back(Move{x, y, x - 1, y - 2});
    }
    if (board.pieceidOn(x + 1, y) == EMPTY_PIECEID)
    {
        mayAvailableMoves.emplace_back(Move{x, y, x + 2, y + 1});
        mayAvailableMoves.emplace_back(Move{x, y, x + 2, y - 1});
    }
    if (board.pieceidOn(x - 1, y) == EMPTY_PIECEID)
    {
        mayAvailableMoves.emplace_back(Move{x, y, x - 2, y + 1});
        mayAvailableMoves.emplace_back(Move{x, y, x - 2, y - 1});
    }

    for (Move v : mayAvailableMoves)
    {
        TEAM targetTeam = board.teamOn(v.x2, v.y2);
        if (targetTeam != team && targetTeam != OVERFLOW_TEAM)
        {
            result.emplace_back(v);
        }
    }

    return result;
}

/// @brief 生成车的着法
MOVES Moves::rook(TEAM team, Board &board, int x, int y)
{
    MOVES result{};

    for (int _y = y + 1; _y <= 9; _y++)
    {
        if (board.teamOn(x, _y) == EMPTY_TEAM)
        {
            result.emplace_back(Move{x, y, x, _y});
        }
        else if (board.teamOn(x, _y) != team)
        {
            result.emplace_back(Move{x, y, x, _y});
            break;
        }
        else
        {
            break;
        }
    }
    for (int _y = y - 1; _y >= 0; _y--)
    {
        if (board.teamOn(x, _y) == EMPTY_TEAM)
        {
            result.emplace_back(Move{x, y, x, _y});
        }
        else if (board.teamOn(x, _y) != team)
        {
            result.emplace_back(Move{x, y, x, _y});
            break;
        }
        else
        {
            break;
        }
    }
    for (int _x = x + 1; _x <= 8; _x++)
    {
        if (board.teamOn(_x, y) == EMPTY_TEAM)
        {
            result.emplace_back(Move{x, y, _x, y});
        }
        else if (board.teamOn(_x, y) != team)
        {
            result.emplace_back(Move{x, y, _x, y});
            break;
        }
        else
        {
            break;
        }
    }
    for (int _x = x - 1; _x >= 0; _x--)
    {
        if (board.teamOn(_x, y) == EMPTY_TEAM)
        {
            result.emplace_back(Move{x, y, _x, y});
        }
        else if (board.teamOn(_x, y) != team)
        {
            result.emplace_back(Move{x, y, _x, y});
            break;
        }
        else
        {
            break;
        }
    }

    return result;
}

/// @brief 生成炮的着法
MOVES Moves::cannon(TEAM team, Board &board, int x, int y)
{
    MOVES result{};

    for (int _x = x + 1; _x <= 8; _x++)
    {
        if (board.pieceidOn(_x, y) != EMPTY_PIECEID)
        {
            for (int _x2 = _x + 1; _x2 <= 8; _x2++)
            {
                if (board.teamOn(_x2, y) == EMPTY_PIECEID)
                {
                    continue;
                }
                else if (board.teamOn(_x2, y) != team)
                {
                    result.emplace_back(Move{x, y, _x2, y});
                    break;
                }
                else
                {
                    break;
                }
            }
            break;
        }
        else
        {
            result.emplace_back(Move{x, y, _x, y});
        }
    }
    for (int _x = x - 1; _x >= 0; _x--)
    {
        if (board.pieceidOn(_x, y) != EMPTY_PIECEID)
        {
            for (int _x2 = _x - 1; _x2 >= 0; _x2--)
            {
                if (board.teamOn(_x2, y) == EMPTY_PIECEID)
                {
                    continue;
                }
                else if (board.teamOn(_x2, y) != team)
                {
                    result.emplace_back(Move{x, y, _x2, y});
                    break;
                }
                else
                {
                    break;
                }
            }
            break;
        }
        else
        {
            result.emplace_back(Move{x, y, _x, y});
        }
    }
    for (int _y = y + 1; _y <= 9; _y++)
    {
        if (board.pieceidOn(x, _y) != EMPTY_PIECEID)
        {
            for (int _y2 = _y + 1; _y2 <= 9; _y2++)
            {
                if (board.teamOn(x, _y2) == EMPTY_PIECEID)
                {
                    continue;
                }
                else if (board.teamOn(x, _y2) != team)
                {
                    result.emplace_back(Move{x, y, x, _y2});
                    break;
                }
                else
                {
                    break;
                }
            }
            break;
        }
        else
        {
            result.emplace_back(Move{x, y, x, _y});
        }
    }
    for (int _y = y - 1; _y >= 0; _y--)
    {
        if (board.pieceidOn(x, _y) != EMPTY_PIECEID)
        {
            for (int _y2 = _y - 1; _y2 >= 0; _y2--)
            {
                if (board.teamOn(x, _y2) == EMPTY_PIECEID)
                {
                    continue;
                }
                else if (board.teamOn(x, _y2) != team)
                {
                    result.emplace_back(Move{x, y, x, _y2});
                    break;
                }
                else
                {
                    break;
                }
            }
            break;
        }
        else
        {
            result.emplace_back(Move{x, y, x, _y});
        }
    }

    return result;
}

/// @brief 生成兵的着法
MOVES Moves::pawn(TEAM team, Board &board, int x, int y)
{
    if (team == RED)
    {
        if (y >= 0 && y <= 4)
        {
            if (board.teamOn(x, y + 1) != team)
            {
                return MOVES{Move{x, y, x, y + 1}};
            }
        }
        else
        {
            MOVES result{};
            if (board.teamOn(x, y + 1) != team)
            {
                if (y + 1 <= 9)
                {
                    result.emplace_back(Move{x, y, x, y + 1});
                }
            }
            if (board.teamOn(x + 1, y) != team)
            {
                if (x + 1 <= 8)
                {

                    result.emplace_back(Move{x, y, x + 1, y});
                }
            }
            if (board.teamOn(x - 1, y) != team)
            {
                if (x - 1 >= 0)
                {
                    result.emplace_back(Move{x, y, x - 1, y});
                }
            }
            return result;
        }
    }
    else
    {
        if (y >= 5 && y <= 9)
        {
            if (board.teamOn(x, y - 1) != team)
            {
                return MOVES{Move{x, y, x, y - 1}};
            }
        }
        else
        {
            MOVES result{};
            if (board.teamOn(x, y - 1) != team)
            {
                if (y - 1 >= 0)
                {
                    result.emplace_back(Move{x, y, x, y - 1});
                }
            }
            if (board.teamOn(x + 1, y) != team)
            {
                if (x + 1 <= 8)
                {

                    result.emplace_back(Move{x, y, x + 1, y});
                }
            }
            if (board.teamOn(x - 1, y) != team)
            {
                if (x - 1 >= 0)
                {
                    result.emplace_back(Move{x, y, x - 1, y});
                }
            }
            return result;
        }
    }
    return MOVES{};
}

/// @brief 生成着法
MOVES Moves::generateMoves(Board &board, int x, int y)
{
    PIECEID chessid = board.pieceidOn(x, y);
    TEAM team = board.teamOn(x, y);

    if (chessid == R_KING || chessid == B_KING)
    {
        return Moves::king(team, board, x, y);
    }
    else if (chessid == R_GUARD || chessid == B_GUARD)
    {
        return Moves::guard(team, board, x, y);
    }
    else if (chessid == R_BISHOP || chessid == B_BISHOP)
    {
        return Moves::bishop(team, board, x, y);
    }
    else if (chessid == R_KNIGHT || chessid == B_KNIGHT)
    {
        return Moves::knight(team, board, x, y);
    }
    else if (chessid == R_ROOK || chessid == B_ROOK)
    {
        return Moves::rook(team, board, x, y);
    }
    else if (chessid == R_CANNON || chessid == B_CANNON)
    {
        return Moves::cannon(team, board, x, y);
    }
    else if (chessid == R_PAWN || chessid == B_PAWN)
    {
        return Moves::pawn(team, board, x, y);
    }
    else
    {
        std::cerr << "Invalid chess id: " + std::to_string(chessid) << std::endl;
        return MOVES{};
    }
}

/// @brief 获取当前队伍所有可行着法
/// @param team
/// @return
MOVES Moves::getMoves(Board &board)
{
    if (!board.isKingLive(board.team))
    {
        return MOVES{};
    }

    MOVES result{};

    std::vector<Piece> pieces = board.getPiecesByTeam(board.team);
    for (const Piece &piece : pieces)
    {
        std::vector<Move> moves = Moves::generateMoves(board, piece.x, piece.y);
        for (Move move : moves)
        {
            result.emplace_back(move);
        }
    }

    return result;
}

/// @brief 获取当前队伍所有吃子着法（暂时写成这样，后续优化）
/// @param board
/// @return
MOVES Moves::getCaptrueMoves(Board &board)
{
    MOVES result{};
    MOVES moves = Moves::getMoves(board);
    for (const Move &move : moves)
    {
        if (board.pieceidOn(move.x2, move.y2) != EMPTY_PIECEID)
        {
            result.emplace_back(move);
        }
    }
    return result;
}

/// @brief  获取当前队伍所有好的吃子着法（MVV/LVA）（暂时写成这样，后续优化）
/// @param board
/// @return
MOVES Moves::getGoodCaptures(Board &board)
{
    // SEE
    MOVES result{};
    MOVES moves = Moves::getCaptrueMoves(board);
    const std::map<PIECEID, int> weightPairs{
        {R_ROOK, 4},
        {R_CANNON, 3},
        {R_KNIGHT, 3},
        {R_BISHOP, 2},
        {R_GUARD, 2},
        {R_PAWN, 1},
        {R_KING, 1}};
    std::map<int, MOVES> orderMap{};

    for (const Move &move : moves)
    {
        int score = 0;

        Piece attacker = board.piecePosition(move.x1, move.y1);
        Piece captured = board.piecePosition(move.x2, move.y2);
        int a = weightPairs.at(abs(captured.pieceid));
        int b = weightPairs.at(abs(attacker.pieceid));
        if (relationship_hasProtector(board, captured.x, captured.y))
        {
            score = a - b + 1;
        }
        else
        {
            score = a + 1;
        }
        if (score < 0)
        {
            PIECEID pieceid = abs(captured.pieceid);
            if (pieceid == R_KNIGHT || pieceid == R_CANNON || pieceid == R_ROOK)
            {
                score = 1;
            }
            if (isGoodPawn(board, captured.x, captured.y))
            {
                score = 1;
            }
        }
        orderMap[score].emplace_back(move);
    }

    for (int score = 8; score > 1; score--)
    {
        for (const Move &move : orderMap[score])
        {
            result.emplace_back(move);
        }
    }

    return result;
}
