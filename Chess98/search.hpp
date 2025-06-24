#pragma once
#include "moves.hpp"
#include "heuristic.hpp"
#include "utils.hpp"
#include "book.hpp"

class Search
{
public:
    Search() = default;
    ~Search()
    {
        delete pHistory;
        delete pKiller;
        delete pTransportation;
    }
    void reset(Board &board)
    {
        this->rootMoves = MOVES{};
        board.distance = 0;
        board.initEvaluate();
        this->pHistory->reset();
        this->pKiller->reset();
        this->pTransportation->reset();
        board = Board::reset(board);
    }

    Result searchMain(Board &board, int maxDepth, int maxTime);
    Result searchOpenBook(Board &board);
    Result searchRoot(Board &board, int depth);
    int searchPV(Board &board, int depth, int alpha, int beta);
    int searchCut(Board &board, int depth, int beta, bool banNullMove = false);
    int searchQ(Board &board, int alpha, int beta, int maxDistance = MAX_SEARCH_DISTANCE);

    MOVES rootMoves{};
    HistoryHeuristic *pHistory = new HistoryHeuristic{};
    KillerTable *pKiller = new KillerTable{};
    TransportationTable *pTransportation = new TransportationTable{};
};

// tricks

void avoidInvalidMoves(Board &board, bool mChecking, MOVES &availableMoves)
{
    if (mChecking)
    {
        MOVES moves{};
        for (const Move &move : availableMoves)
        {
            Piece eaten = board.doMove(move);
            board.team = -board.team;
            if (inCheck(board) == false || board.isKingLive(-board.team) == false)
            {
                moves.emplace_back(move);
            }
            board.team = -board.team;
            board.undoMove(move, eaten);
        }
        availableMoves = moves;
    }
}

void setCheckingMove(Board &board, bool mChecking)
{
    if (mChecking && board.historyMoves.size() > 0)
    {
        board.historyMoves.back().isCheckingMove = true;
    }
}

TrickResult<int> nullAndDeltaPruning(Board &board, bool mChecking, int &alpha, int &beta, int &vlBest)
{
    if (!mChecking)
    {
        int vl = board.evaluate();
        if (vl >= beta)
        {
            return TrickResult<int>{true, {vl}};
        }
        // delta pruning
        if (vl <= alpha - DELTA_PRUNING_MARGIN)
        {
            return TrickResult<int>{true, {alpha}};
        }
        vlBest = vl;
        if (vl > alpha)
        {
            alpha = vl;
        }
    }
    return TrickResult<int>{false, {}};
}

TrickResult<int> mateDistancePruning(Board &board, int alpha, int &beta)
{
    const int vlDistanceMate = INF - board.distance;
    if (vlDistanceMate < beta)
    {
        beta = vlDistanceMate;
        if (alpha >= vlDistanceMate)
        {
            return TrickResult<int>(true, {vlDistanceMate});
        }
    }
    return TrickResult<int>(false, {});
}

TrickResult<int> futilityPruning(Board &board, int beta, int depth)
{
    if (depth == 1)
    {
        int vl = board.evaluate();
        if (vl <= beta - FUTILITY_PRUNING_MARGIN)
        {
            return TrickResult<int>{true, {vl}};
        }
        if (vl >= beta + FUTILITY_PRUNING_MARGIN)
        {
            return TrickResult<int>{true, {vl}};
        }
    }
    return TrickResult<int>{false, {}};
}

TrickResult<int> mutiProbcut(Board &board, Search *search, SEARCH_TYPE searchType, int alpha, int beta, int depth)
{
    if (depth % 4 == 0)
    {
        const double vlScale = (double)vlPawn / 100.0;
        const double a = 1.02 * vlScale;
        const double b = 2.36 * vlScale;
        const double sigma = 82.0 * vlScale;
        const double t = 1.5;
        const int upperBound = int((t * sigma + beta - b) / a);
        const int lowerBound = int((-t * sigma + alpha - b) / a);
        if (search->searchCut(board, depth - 2, upperBound) >= upperBound)
        {
            return TrickResult<int>{true, {beta}};
        }
        else if (search->searchCut(board, depth - 2, lowerBound + 1) <= lowerBound && searchType == PV)
        {
            return TrickResult<int>{true, {alpha}};
        }
    }

    return TrickResult<int>{false, {}};
}

// functions

Result Search::searchMain(Board &board, int maxDepth, int maxTime = 3)
{
    if (board.isKingLive(RED) == false || board.isKingLive(BLACK) == false)
    {
        exit(0);
    }

    // openbook search
    Result openbookResult = Search::searchOpenBook(board);
    if (openbookResult.val != -1)
    {
        std::cout << "Find a great move from OpenBook!" << std::endl;
        return openbookResult;
    }

    this->reset(board);

    std::cout << board.evaluate() << std::endl;

    this->rootMoves = Moves::getMoves(board);
    
    Result bestNode = Result(Move(), 0);
    clock_t start = clock();

    for (int depth = 0; depth <= maxDepth; depth++)
    {
        bestNode = searchRoot(board, depth);
        // log
        std::cout << "depth: " << depth + 1;
        std::cout << " | vl: " << bestNode.val;
        std::cout << " | moveid: " << bestNode.move.id;
        std::cout << " | duration(ms): " << clock() - start << std::endl;
        // timeout break
        if (clock() - start >= maxTime * 1000 / 3)
        {
            break;
        }
    }

    return bestNode;
}

Result Search::searchOpenBook(Board &board)
{
    BookStruct bk;
    BookFileStruct *pBookFileStruct = new BookFileStruct{};

    if (!pBookFileStruct->open("BOOK.DAT"))
    {
        delete pBookFileStruct;
        return Result{Move{}, -1};
    }

    // 二分法查找开局库
    int nMid = 0;
    int32 hashLock = board.hashLock;
    int32 mirrorHashLock = 0;
    int32 mirrorHashKey = 0;
    board.getMirrorHashinfo(mirrorHashKey, mirrorHashLock);

    int nScan = 0;
    int32 nowHashLock = 0;
    for (nScan = 0; nScan < 2; nScan++)
    {
        int nHigh = pBookFileStruct->nLen - 1;
        int nLow = 0;
        nowHashLock = (nScan == 0) ? hashLock : mirrorHashLock;
        while (nLow <= nHigh)
        {
            nMid = (nHigh + nLow) / 2;
            pBookFileStruct->read(bk, nMid);
            if (BOOK_POS_CMP(bk, nowHashLock) < 0)
            {
                nLow = nMid + 1;
            }
            else if (BOOK_POS_CMP(bk, nowHashLock) > 0)
            {
                nHigh = nMid - 1;
            }
            else
            {
                break;
            }
        }
        if (nLow <= nHigh)
        {
            break;
        }
    }

    if (nScan == 2)
    {
        pBookFileStruct->close();
        return Result{Move{}, -1};
    }

    // 如果找到局面，则向前查找第一个着法
    for (nMid--; nMid >= 0; nMid--)
    {
        pBookFileStruct->read(bk, nMid);
        if (BOOK_POS_CMP(bk, nowHashLock) < 0)
        {
            break;
        }
    }

    std::vector<Move> bookMoves;

    // 向后依次读入属于该局面的每个着法
    for (nMid++; nMid < pBookFileStruct->nLen; nMid++)
    {
        pBookFileStruct->read(bk, nMid);
        if (BOOK_POS_CMP(bk, nowHashLock) > 0)
        {
            break;
        }
        else
        {
            int mv = bk.wmv;
            int src = mv & 255;
            int dst = mv >> 8;
            int xSrc = (src & 15) - 3;
            int ySrc = 12 - (src >> 4);
            int xDst = (dst & 15) - 3;
            int yDst = 12 - (dst >> 4);
            if (nScan != 0)
            {
                xSrc = 8 - xSrc;
                xDst = 8 - xDst;
            }
            int vl = bk.wvl;
            Move tMove = Move(xSrc, ySrc, xDst, yDst, vl);
            bookMoves.emplace_back(tMove);
        }
    }

    // 从大到小排序
    std::sort(bookMoves.begin(), bookMoves.end(),
              [](Move &a, Move &b)
              { return a.val > b.val; });

    std::srand(unsigned(std::time(0)));

    int vlSum = 0;
    for (Move &move : bookMoves)
    {
        vlSum += move.val;
    }
    int vlRandom = std::rand() % vlSum;

    Move bookMove;
    for (Move &move : bookMoves)
    {
        vlRandom -= move.val;
        if (vlRandom < 0)
        {
            bookMove = move;
            break;
        }
    }

    pBookFileStruct->close();

    delete pBookFileStruct;

    return Result{bookMove, 1};
}

Result Search::searchRoot(Board &board, int depth)
{
    Move bestMove{};
    int vl = -INF;
    int vlBest = -INF;
    if (board.historyMoves.size() > 4)
    {
        Move lastMove = board.historyMoves.back();
        board.undoMove(lastMove, lastMove.captured);
        if (board.isRepeatStatus())
        {
            std::cout << "e" << std::endl;
        }

        board.doMove(lastMove);
    }
    // 若检测到被将军则避免送将着法
    avoidInvalidMoves(board, inCheck(board), rootMoves);

    for (const Move &move : rootMoves)
    {
        Piece eaten = board.doMove(move);

        // 避免重复局面
        if (board.isRepeatStatus())
        {
            board.undoMove(move, eaten);
            continue;
        }

        if (vlBest == -INF)
        {
            vl = -searchPV(board, depth - 1, -INF, -vlBest);
        }
        else
        {
            vl = -searchCut(board, depth - 1, -vlBest);
            if (vl > vlBest)
            {
                vl = -searchPV(board, depth - 1, -INF, -vlBest);
            }
        }
        if (vl > vlBest)
        {
            vlBest = vl;
            bestMove = move;
            // search step
            for (Move &move : rootMoves)
            {
                if (bestMove == move)
                {
                    move.val = INF;
                }
                else
                {
                    move.val--;
                }
            }
        }

        board.undoMove(move, eaten);
    }

    if (bestMove.id == -1)
    {
        vlBest += board.distance;
    }
    else
    {
        this->pHistory->add(bestMove, depth);
        this->pTransportation->add(board, bestMove, vlBest, EXACT_TYPE, depth);
    }

    // 历史启发调整根节点似乎更快
    this->pHistory->sort(rootMoves);

    if (bestMove.id != -1)
    {
        return Result{bestMove, vlBest};
    }
    else
    {
        return Result{Moves::getMoves(board)[0], vlBest};
    }
}

// core

int Search::searchPV(Board &board, int depth, int alpha, int beta)
{
    // 检查将帅是否在棋盘上
    if (board.isKingLive(board.team) == false || board.isKingLive(-board.team) == false)
    {
        return board.isKingLive(board.team) == false ? -INF : INF;
    }
    if (board.isRepeatStatus())
    {
        return INF * 2;
    }

    // 置换表分数
    const int vlHash = this->pTransportation->getValue(board, alpha, beta, depth);
    if (vlHash != -INF)
    {
        if (vlHash >= beta)
        {
            if (Search::searchQ(board, beta - 1, beta, board.distance + QUIESCENCE_EXTEND_DEPTH) >= beta)
            {
                return vlHash;
            }
            else if (depth <= 0)
            {
                return beta;
            }
        }
        else if (vlHash <= alpha)
        {
            if (Search::searchQ(board, alpha, alpha + 1, board.distance + QUIESCENCE_EXTEND_DEPTH) <= alpha)
            {
                return vlHash;
            }
            else if (depth <= 0)
            {
                return alpha;
            }
        }
        else
        {
            const int vl1 = Search::searchQ(board, alpha, alpha + 1, board.distance + QUIESCENCE_EXTEND_DEPTH);
            const int vl2 = Search::searchQ(board, beta - 1, beta, board.distance + QUIESCENCE_EXTEND_DEPTH);
            if (vl1 > alpha && vl2 < beta)
            {
                return vlHash;
            }
        }
    }

    // 静态搜索
    if (depth <= 0)
    {
        int vl = Search::searchQ(board, alpha, beta, board.distance + QUIESCENCE_EXTEND_DEPTH);
        this->pTransportation->add(board, Move{}, vl, EXACT_TYPE, depth);
        return vl;
    }

    // mate distance pruning
    TrickResult<int> result = mateDistancePruning(board, alpha, beta);
    if (result.isSuccess)
    {
        return result.data[0];
    }

    // variables
    const bool mChecking = inCheck(board);

    // 验证上一步是否是将军着法
    setCheckingMove(board, mChecking);

    // tricks
    if (!mChecking)
    {
        // futility pruning
        TrickResult<int> futilityResult = futilityPruning(board, beta, depth);
        if (futilityResult.isSuccess)
        {
            return futilityResult.data[0];
        }

        // multi probCut
        TrickResult<int> probCutResult = mutiProbcut(board, this, PV, alpha, beta, depth);
        if (probCutResult.isSuccess)
        {
            return probCutResult.data[0];
        }
    }

    // variables
    int vlBest = -INF;
    Move bestMove{};
    NODE_TYPE type = ALPHA_TYPE;

    // 置换表着法
    Move goodMove = this->pTransportation->getMove(board);
    if (goodMove.id == -1 && depth >= 2)
    {
        if (searchPV(board, depth / 2, alpha, beta) <= alpha)
        {
            searchPV(board, depth / 2, -INF, beta);
        }
        goodMove = this->pTransportation->getMove(board);
    }
    if (goodMove.id != -1)
    {
        Piece eaten = board.doMove(goodMove);
        vlBest = -searchPV(board, depth - 1, -beta, -alpha);
        board.undoMove(goodMove, eaten);
        bestMove = goodMove;
        if (vlBest >= beta)
        {
            type = BETA_TYPE;
        }
        if (vlBest > alpha)
        {
            type = EXACT_TYPE;
            alpha = vlBest;
        }
    }

    // 搜索
    if (type != BETA_TYPE)
    {
        int vl = -INF;
        MOVES availableMoves = Moves::getMoves(board);

        // 若检测到被将军则避免送将着法
        avoidInvalidMoves(board, mChecking, availableMoves);

        // 历史启发
        this->pHistory->sort(availableMoves);

        for (const Move &move : availableMoves)
        {
            Piece eaten = board.doMove(move);

            // 避免重复局面
            if (board.isRepeatStatus())
            {
                board.undoMove(move, eaten);
                continue;
            }

            if (vlBest == -INF)
            {
                vl = -searchPV(board, depth - 1, -beta, -alpha);
            }
            else
            {
                vl = -searchCut(board, depth - 1, -alpha);
                if (vl > alpha && vl < beta)
                {
                    vl = -searchPV(board, depth - 1, -beta, -alpha);
                }
            }

            board.undoMove(move, eaten);

            // 更新最佳值
            if (vl > vlBest)
            {
                vlBest = vl;
                bestMove = move;
                if (vl >= beta)
                {
                    type = BETA_TYPE;
                    break;
                }
                if (vl > alpha)
                {
                    type = EXACT_TYPE;
                    alpha = vl;
                }
            }
        }
    }

    // 结果
    if (bestMove.id == -1)
    {
        vlBest += board.distance;
    }
    else if (type != ALPHA_TYPE)
    {
        this->pHistory->add(bestMove, depth);
        this->pTransportation->add(board, bestMove, vlBest, type, depth);
    }

    return vlBest;
}

int Search::searchCut(Board &board, int depth, int beta, bool banNullMove)
{
    if (board.isRepeatStatus())
    {
        return INF * 2;
    }
    // 检查将帅是否在棋盘上
    if (board.isKingLive(board.team) == false || board.isKingLive(-board.team) == false)
    {
        return board.isKingLive(board.team) == false ? -INF : INF;
    }

    // 置换表分数
    int vlHash = this->pTransportation->getValue(board, beta - 1, beta, depth);
    if (vlHash != -INF)
    {
        int statisValue = Search::searchQ(board, beta - 1, beta, board.distance + QUIESCENCE_EXTEND_DEPTH);
        if (vlHash >= beta && statisValue >= beta && board.distance + QUIESCENCE_EXTEND_DEPTH)
        {
            return vlHash;
        }
        else if (vlHash < beta && statisValue < beta)
        {
            return vlHash;
        }
        else if (depth <= 0)
        {
            return statisValue;
        }
    }

    // 静态搜索
    if (depth <= 0)
    {
        return Search::searchQ(board, beta - 1, beta, board.distance + QUIESCENCE_EXTEND_DEPTH);
    }

    // mate distance pruning
    TrickResult<int> trickresult = mateDistancePruning(board, beta - 1, beta);
    if (trickresult.isSuccess)
    {
        return trickresult.data[0];
    }

    // variables
    const bool mChecking = inCheck(board);

    // 验证上一步是否是将军着法
    setCheckingMove(board, mChecking);

    // tricks
    if (!mChecking)
    {
        // futility pruning
        TrickResult<int> futilityResult = futilityPruning(board, beta, depth);
        if (futilityResult.isSuccess)
        {
            return futilityResult.data[0];
        }
        // multi probCut and null pruning
        if (!banNullMove)
        {
            if (board.nullOkay())
            {
                board.doNullMove();
                int vl = -searchCut(board, depth - 3, -beta + 1, true);
                board.undoNullMove();
                if (vl >= beta)
                {
                    if (board.nullSafe())
                    {
                        return vl;
                    }
                    else if (searchCut(board, depth - 2, beta, true) >= beta)
                    {
                        return vl;
                    }
                }
            }
        }
        else
        {
            TrickResult<int> probCutResult = mutiProbcut(board, this, CUT, 0, beta, depth);
            if (probCutResult.isSuccess)
            {
                return probCutResult.data[0];
            }
        }
    }

    // variables
    int vlBest = -INF;
    Move bestMove{};
    NODE_TYPE type = ALPHA_TYPE;
    int searchedCnt = 0;

    // 置换表着法
    Move goodMove = this->pTransportation->getMove(board);
    if (goodMove.id != -1)
    {
        Piece eaten = board.doMove(goodMove);
        int vl = -searchCut(board, depth - 1, -beta + 1);
        board.undoMove(goodMove, eaten);
        bestMove = goodMove;
        if (vl > vlBest)
        {
            vlBest = vl;
            if (vl >= beta)
            {
                type = BETA_TYPE;
            }
        }
    }

    // 杀手启发
    if (type != BETA_TYPE)
    {
        MOVES killerAvailableMoves = this->pKiller->get(board);
        MOVES _moves = Moves::getMoves(board);
        for (const Move &move : killerAvailableMoves)
        {
            Piece eaten = board.doMove(move);
            int vl = -searchCut(board, depth - 1, -beta + 1);
            board.undoMove(move, eaten);
            if (vl > vlBest)
            {
                vlBest = vl;
                bestMove = move;
                if (vl >= beta)
                {
                    type = BETA_TYPE;
                    break;
                }
            }
        }
    }

    // 搜索
    if (type != BETA_TYPE)
    {
        // 获取所有可行着法
        MOVES availableMoves = Moves::getMoves(board);

        // 若检测到被将军则避免送将着法
        avoidInvalidMoves(board, mChecking, availableMoves);

        // 历史启发
        this->pHistory->sort(availableMoves);

        for (Move &move : availableMoves)
        {
            Piece eaten = board.doMove(move);
            int vl = -INF;

            // 避免重复局面
            if (board.isRepeatStatus())
            {
                board.undoMove(move, eaten);
                continue;
            }

            // lmr pruning
            if (!mChecking &&
                eaten.pieceid == EMPTY_PIECEID &&
                depth >= 3 &&
                searchedCnt >= 4)
            {
                vl = -searchCut(board, depth - 2 - static_cast<int>(depth >= 4), -beta + 1);
            }
            else
            {
                vl = -searchCut(board, depth - 1, -beta + 1);
            }

            board.undoMove(move, eaten);

            // 更新最佳值
            if (vl > vlBest)
            {
                vlBest = vl;
                bestMove = move;
                if (vl >= beta)
                {
                    type = BETA_TYPE;
                    break;
                }
            }

            searchedCnt++;
        }
    }

    // 结果
    if (bestMove.id == -1)
    {
        vlBest += board.distance;
    }
    else if (type == BETA_TYPE)
    {
        this->pHistory->add(bestMove, depth);
        this->pKiller->add(board, bestMove);
        this->pTransportation->add(board, bestMove, vlBest, type, depth);
    }

    return vlBest;
}

int Search::searchQ(Board &board, int alpha, int beta, int maxDistance)
{
    // 检测将帅是否在棋盘上
    if (board.isKingLive(board.team) == false || board.isKingLive(-board.team) == false)
    {
        return board.isKingLive(board.team) == false ? -INF : INF;
    }

    // 返回评估结果
    if (board.distance > maxDistance || true)
    {
        return board.evaluate();
    }

    // mate distance pruning
    TrickResult<int> trickresult = mateDistancePruning(board, alpha, beta);
    if (trickresult.isSuccess)
    {
        return trickresult.data[0];
    }

    // variables
    const bool mChecking = inCheck(board);
    int vlBest = -INF;

    // 验证上一步是否是将军着法
    setCheckingMove(board, mChecking);

    // null and delta pruning
    TrickResult<int> nullDeltaResult = nullAndDeltaPruning(board, mChecking, alpha, beta, vlBest);
    if (nullDeltaResult.isSuccess)
    {
        return nullDeltaResult.data[0];
    }

    // 搜索
    MOVES availableMoves = mChecking ? Moves::getMoves(board) : Moves::getCaptureMoves(board);

    for (const Move &move : availableMoves)
    {
        Piece eaten = board.doMove(move);

        // 避免重复局面
        if (board.isRepeatStatus())
        {
            board.undoMove(move, eaten);
            continue;
        }

        int vl = -Search::searchQ(board, -beta, -alpha, maxDistance - 1);
        board.undoMove(move, eaten);
        if (vl > vlBest)
        {
            if (vl >= beta)
            {
                return vl;
            }
            vlBest = vl;
            if (vl > alpha)
            {
                alpha = vl;
            }
        }
    }

    // 结果
    if (vlBest == -INF)
    {
        vlBest += board.distance;
    }

    return vlBest;
}
