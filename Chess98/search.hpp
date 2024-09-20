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
    static Node minmax(Board board, int depth, bool isMax);
};

/// @brief minmax搜索算法
/// @param depth 深度
/// @param isMax 节点类型，true为max节点，false为min节点
/// @return 节点
Node Searcher::minmax(Board board, int depth, bool isMax)
{
    TEAM team = isMax ? RED : BLACK;
    MOVES availableMoves = MovesGenerator::getMovesOf(board, team);

    std::vector<int> scores{};
    std::vector<Move> moves{};

    for (Move move : availableMoves)
    {
        Piece eaten = board.doMove(move);

        if (depth > 0)
        {
            Node node = minmax(board, depth - 1, !isMax);
            scores.emplace_back(node.score);
            moves.emplace_back(node.move);
        }
        else
        {
            int score = Evaluator::evaluate(board);
            scores.emplace_back(score);
            moves.emplace_back(move);
        }

        board.undoMove(move, eaten);
    }

    int index = std::max_element(scores.begin(), scores.end()) - scores.begin();
    Node node{ moves[index], scores[index] };

    return node;
}
