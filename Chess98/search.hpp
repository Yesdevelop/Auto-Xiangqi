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
        if (historyCache)
        {
            delete historyCache;
            historyCache = nullptr;
        }
        if (pKillerTable)
        {
            delete pKillerTable;
            pKillerTable = nullptr;
        }
        if (pHashTable)
        {
            delete pHashTable;
            pHashTable = nullptr;
        }
        if (pBookFileStruct)
        {
            delete pBookFileStruct;
            pBookFileStruct = nullptr;
        }
    }

    Result searchMain(Board &board, int maxDepth, int maxTime);
    Result searchOpenBook(Board &board);
    Result searchRoot(Board &board, int depth);
    int searchPV(Board &board, int depth, int alpha, int beta);
    int searchCut(Board &board, int depth, int beta, bool banNullMove = false);
    int searchQ(Board &board, int alpha, int beta, int maxDistance = maxSearchDistance);

private:
    /// @brief 初始化搜索
    /// @param board
    /// @param initHashLevel
    void searchInit(Board &board, int initHashLevel = 25)
    {
        rootMoves.resize(0);
        this->historyCache->init();
        if (this->pKillerTable->initDone())
        {
            this->pKillerTable->reset();
        }
        else
        {
            this->pKillerTable->init();
        }
        if (this->pHashTable->initDone())
        {
            this->pHashTable->reset();
        }
        else
        {
            this->pHashTable->init(initHashLevel);
        }
        board.distance = 0;
        board.initEvaluate();
    }

    void searchStep(Move &bestMove)
    {
        for (auto &move : rootMoves)
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

    /// @brief 根节点着法排序
    void sortRootMoves()
    {
        std::sort(
            rootMoves.begin(), rootMoves.end(),
            [](Move &first, Move &second) -> bool
            {
                return first.val > second.val;
            });
    }

    MOVES rootMoves;

    HistoryHeuristic *historyCache = new HistoryHeuristic{};
    KillerTable* pKillerTable = new KillerTable{};
    TransportationTable*pHashTable = new TransportationTable{};
    BookFileStruct *pBookFileStruct = new BookFileStruct{};
};

/// @brief 迭代加深
/// @param board
/// @param maxDepth
/// @param maxTime
/// @return
Result Search::searchMain(Board &board, int maxDepth, int maxTime = 3)
{
    if (board.isKingLive(RED) == false || board.isKingLive(BLACK) == false)
        exit(0);

    std::cout << "---------------------" << std::endl;

    // 开局库搜索
#ifndef BAN_OPENBOOK
    Result openbookResult = Search::searchOpenBook(board);
    if (openbookResult.val != -1)
    {
        std::cout << "Find a great move from OpenBook!" << std::endl;
        return openbookResult;
    }
#endif

    this->searchInit(board);
    this->rootMoves = Moves::getMoves(board);

    Result bestNode = Result(Move(), 0);
    clock_t start = clock();

    for (int depth = 0; depth <= maxDepth; depth++)
    {
        bestNode = searchRoot(board, depth);
        std::cout << "depth: " << depth + 1;
        std::cout << " | vl: " << bestNode.val;
        std::cout << " | moveid: " << bestNode.move.id;
        std::cout << " | duration(ms): " << clock() - start << std::endl;
        // 杀棋中止
        if (std::abs(bestNode.val) >= BAN)
        {
            break;
        }
        // 超时中止
        else if (clock() - start >= maxTime * 1000 / 3)
        {
            break;
        }
    }

    Piece eaten = board.doMove(bestNode.move);
    if (inCheck(board) == true)
        bestNode.move.isCheckingMove = true;
    board.undoMove(bestNode.move, eaten);

    return bestNode;
}

/// @brief 搜索开局库
/// @param board
Result Search::searchOpenBook(Board &board)
{
    BookStruct bk;
    if (!pBookFileStruct->open("BOOK.DAT"))
    {
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

    return Result{bookMove, 1};
}

/// @brief 根节点搜索
/// @param board
/// @param depth
/// @return
Result Search::searchRoot(Board &board, int depth)
{
    Move *pBestMove = nullptr;
    int vl = -INF;
    int vlBest = -INF;

    for (Move &move : rootMoves)
    {
        Piece eaten = board.doMove(move);

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

        board.undoMove(move, eaten);

        if (vl > vlBest)
        {
            vlBest = vl;
            pBestMove = &move;
            this->searchStep(move);
        }
    }

    if (!pBestMove)
    {
        vlBest += board.distance;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
    }
    Result result{!pBestMove ? Move{} : *pBestMove, vlBest};

    sortRootMoves();

    return result;
}

/// @brief PV搜索
/// @param board
/// @param depth
/// @param alpha
/// @param beta
/// @return
int Search::searchPV(Board &board, int depth, int alpha, int beta)
{
    // 叶节点返回静态评估值
    if (depth <= 0)
    {
        return Search::searchQ(board,alpha,beta);
    }

    // mate distance pruning
    const int vlDistanceMate = INF - board.distance;
    if (vlDistanceMate < beta)
    {
        beta = vlDistanceMate;
        if (alpha >= vlDistanceMate)
        {
            return vlDistanceMate;
        }
    }

    const bool mChecking = inCheck(board);

    if (!mChecking)
    {
        // futility pruning
        if (depth == 1)
        {
            int vl = board.evaluate();
            if (vl <= beta - futilityPruningMargin)
            {
                return vl;
            }
            if (vl >= beta + futilityPruningMargin)
            {
                return vl;
            }
        }
        // multi probCut
        if (depth % 4 == 0)
        {
            const double vlScale = (double)vlPawn / 100.0;
            const double a = 1.02 * vlScale;
            const double b = 2.36 * vlScale;
            const double sigma = 82.0 * vlScale;
            const double t = 1.5;
            const int upperBound = int((t * sigma + beta - b) / a);
            const int lowerBound = int((-t * sigma + alpha - b) / a);
            if (searchCut(board, depth - 2, upperBound) >= upperBound)
            {
                return beta;
            }
            else if (searchCut(board, depth - 2, lowerBound + 1) <= lowerBound)
            {
                return alpha;
            }
        }
    }

    int vlBest = -INF;
    Move* pBestMove = nullptr;
    nodeType type = alphaType;
    Move goodMove;
    this->pHashTable->get(board, goodMove);
    if (goodMove.x1 == -1 && depth >= 2)
    {
        if (searchPV(board, depth / 2, alpha, beta) <= alpha)
        {
            searchPV(board, depth / 2, -INF, beta);
        }
        this->pHashTable->get(board, goodMove);
    }

    if (goodMove.x1 != -1)
    {
        Piece eaten = board.doMove(goodMove);
        vlBest = -searchPV(board, depth - 1, -beta, -alpha);
        board.undoMove(goodMove, eaten);
        pBestMove = &goodMove;
        if (vlBest >= beta)
        {
            type = betaType;
        }
        if (vlBest > alpha)
        {
            type = exactType;
            alpha = vlBest;
        }
    }

    MOVES availableMoves;
    if (type != betaType)
    {
        int vl = -INF;
        availableMoves = Moves::getMoves(board);
        this->historyCache->sort(availableMoves);
        for (auto& move : availableMoves)
        {
            Piece eaten = board.doMove(move);
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

            if (vl > vlBest)
            {
                vlBest = vl;
                pBestMove = &move;
                if (vl >= beta)
                {
                    type = betaType;
                    break;
                }
                if (vl > alpha)
                {
                    type = exactType;
                    alpha = vl;
                }
            }
        }
    }

    if (!pBestMove)
    {
        vlBest += board.distance;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
        this->pHashTable->add(board,*pBestMove);
    }

    return vlBest;
}

/// @brief 截断节点搜索
/// @param board
/// @param depth
/// @param beta
/// @return
int Search::searchCut(Board &board, int depth, int beta, bool banNullMove)
{
    if (depth <= 0)
    {
        return Search::searchQ(board,beta - 1, beta);
    }

    // mate distance pruning
    const int vlDistanceMate = INF - board.distance;
    const int vlOriginAlpha = beta - 1;
    if (vlDistanceMate < beta)
    {
        beta = vlDistanceMate;
        if (vlOriginAlpha >= vlDistanceMate)
        {
            return vlDistanceMate;
        }
    }

    const bool mChecking = inCheck(board);

    if (!mChecking)
    {
        // futility pruning
        if (depth == 1)
        {
            int vl = board.evaluate();
            if (vl <= beta - futilityPruningMargin)
            {
                return vl;
            }
            if (vl >= beta + futilityPruningMargin)
            {
                return vl;
            }
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
        else if (depth % 4 == 0)
        {
            const double vlScale = (double)vlPawn / 100.0;
            const double a = 1.02 * vlScale;
            const double b = 2.36 * vlScale;
            const double sigma = 82.0 * vlScale;
            const double t = 1.5;
            const int upperBound = int((t * sigma + beta - b) / a);
            if (searchCut(board, depth - 2, upperBound) >= upperBound)
            {
                return beta;
            }
        }
    }
    nodeType type = alphaType;
    Move* pBestMove = nullptr;
    int vlBest = -INF;
    int searchedCnt = 0;

    MOVES killerAvaliableMoves;
    if (type != betaType)
    {
        killerAvaliableMoves = this->pKillerTable->get(board);
        for (auto& move : killerAvaliableMoves)
        {
            Piece eaten = board.doMove(move);
            int vl = -searchCut(board, depth - 1, -beta + 1);
            board.undoMove(move, eaten);
            if (vl > vlBest)
            {
                vlBest = vl;
                pBestMove = &move;
                if (vl >= beta)
                {
                    type = betaType;
                    break;
                }
            }
        }
    }

    MOVES availableMoves;
    if (type != betaType)
    {
        availableMoves = Moves::getMoves(board);
        this->historyCache->sort(availableMoves);
        for (auto& move : availableMoves)
        {
            Piece eaten = board.doMove(move);
            int vl = -INF;
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
            if (vl > vlBest)
            {
                vlBest = vl;
                pBestMove = &move;
                if (vl >= beta)
                {
                    type = betaType;
                    break;
                }
            }
            searchedCnt++;
        }
    }

    if (!pBestMove)
    {
        vlBest += board.distance;
    }
    else
    {
        this->historyCache->add(*pBestMove, depth);
        this->pKillerTable->add(board, *pBestMove);
    }

    return vlBest;
}

/// @brief 静态搜索函数
/// @param board
/// @param alpha
/// @param beta
/// @return
int Search::searchQ(Board &board, int alpha, int beta, int maxDistance)
{
    if (board.distance >= maxDistance)
    {
        return board.evaluate();
    }

    // mate distance pruning
    const int vlDistanceMate = INF - board.distance;
    if (vlDistanceMate < beta)
    {
        beta = vlDistanceMate;
        if (alpha >= vlDistanceMate)
        {
            return vlDistanceMate;
        }
    }

    // null and delta pruning
    const bool mChecking = inCheck(board);
    int leftDistance = mChecking ? std::min<int>(4, maxDistance - 1) : maxDistance - 1;
    int vlBest = -INF;
    if (!mChecking)
    {
        int vl = board.evaluate();
        if (vl >= beta)
        {
            return vl;
        }
        // delta pruning
        if (vl <= alpha - deltaPruningMargin)
        {
            return alpha;
        }

        vlBest = vl;
        alpha = std::max<int>(alpha, vl);
    }

    MOVES availableMoves = mChecking ? Moves::getMoves(board) : Moves::getCaptureMoves(board);

    for (const Move &move : availableMoves)
    {
        Piece eaten = board.doMove(move);
        int vl = -Search::searchQ(board, -beta, -alpha, leftDistance);
        board.undoMove(move, eaten);
        if (vl > vlBest)
        {
            if (vl >= beta)
            {
                return vl;
            }
            vlBest = vl;
            alpha = std::max<int>(alpha, vl);
        }
    }

    if (vlBest == -INF)
    {
        vlBest += board.distance;
    }

    return vlBest;
}
