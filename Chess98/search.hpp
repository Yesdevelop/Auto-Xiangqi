#include <chrono>
#include "moves.hpp"
#include "evaluate.hpp"

/// @brief 节点对象，存储分数 + 着法
class Node
{
public:
    Node(Move move, int score) : move(move), score(score) {}
    Move move{};
    int score = 0;
};

/// @brief 搜索工具类
class Searcher
{
public:
    static Node search(Board &board, TEAM currentTeam, int time);
    static Node alphabeta(Board &board, int depth, bool isMax, int alpha, int beta);
};

/// @brief 迭代加深搜索
/// @param board
/// @param currentTeam
/// @param depth
/// @return
Node Searcher::search(Board &board, TEAM currentTeam, int time)
{
    Node result{Move{}, 0};
    int depth = 0;

    CHRONO_TIME_T startTime = getCurrentTimeWithMS();
    std::cout << startTime;
    do
    {
        result = Searcher::alphabeta(
            board,
            depth,
            currentTeam == RED ? true : false,
            -100000, 100000);
        depth += 1;
    } while (getCurrentTimeWithMS() - startTime < time);
    
    return result;
}

/// @brief minmax搜索算法
/// @param depth 深度
/// @param isMax 节点类型，true为max节点，false为min节点
/// @return 节点
Node Searcher::alphabeta(Board &board, int depth, bool isMax, int alpha, int beta)
{
    TEAM team = isMax ? RED : BLACK;
    MOVES availableMoves = MovesGenerator::getMovesOf(board, team);

    std::vector<int> scores{};
    std::vector<Move> moves{};

    for (Move move : availableMoves)
    {
        Piece eaten = board.doMove(move);

        Node node{move, 0};
        if (depth > 0)
        {
            Node temp = Searcher::alphabeta(board, depth - 1, !isMax, alpha, beta);
            node.score = temp.score;
        }
        else
        {
            node.score = Evaluator::evaluate(board);
        }
        if (isMax == false && node.score < beta)
        {
            beta = node.score;
        }
        if (isMax == true && node.score > alpha)
        {
            alpha = node.score;
        }
        moves.emplace_back(node.move);
        scores.emplace_back(node.score);

        board.undoMove(move, eaten);

        if (alpha >= beta)
        {
            break;
        }
    }

    if (moves.size() != 0)
    {
        int index = -1;
        if (isMax)
        {
            index = int(std::max_element(scores.begin(), scores.end()) - scores.begin());
        }
        else
        {
            index = int(std::min_element(scores.begin(), scores.end()) - scores.begin());
        }
        Node node{moves[index], scores[index]};

        return node;
    }
    else
    {
        return Node{Move{}, isMax ? -100000 : 100000};
    }
}
