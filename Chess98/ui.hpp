#pragma once
#include "board.hpp"

using BOARD_CODE = std::string;

BOARD_CODE generateCode(Board board)
{
    BOARD_CODE code = "";
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            PIECEID pieceid = board.pieceidOn(i, j);
            std::string name = PIECE_NAME_PAIRS.at(pieceid);
            code += name;
        }
    }
    return code;
}

PIECEID_MAP decode(BOARD_CODE code)
{
    PIECEID_MAP result{};
    for (int i = 0; i < 90; i++)
    {
        size_t x = i / 10;
        size_t y = i % 10;
        char c1 = code[i * size_t(2) - size_t(1)];
        char c2 = code[i * size_t(2)];
        std::string pieceName{c1, c2};
        result[x][y] = NAME_PIECE_PAIRS.at(pieceName);
    }
    return result;
}

void setBoardCode(Board board)
{
    BOARD_CODE code = generateCode(board);

	const std::string historyMovesBack = board.historyMoves.size() > 0 ? std::to_string(board.historyMoves.back().id) : "null";
    const std::string jsPutCode =
        "\
        const http = require('http')\n\
        const options = {\n\
            hostname: '127.0.0.1',\n\
            path: '/?boardcode=" +
        code + "',\n\
            port: 9494,\n\
            method : 'PUT'\n\
        }\n\
        http.request(options).end();\n\
        const options2 = {\n\
            hostname: '127.0.0.1',\n\
            path: '/?move=" +
            historyMovesBack + "',\n\
            port: 9494,\n\
            method : 'PUT'\n\
        }\n\
        http.request(options2).end();\n\
            ";

    FILE *file = nullptr;
    errno_t result = fopen_s(&file, "./_put_.js", "w+");
    if (result == 0)
    {
        fprintf(file, jsPutCode.c_str());
        std::fclose(file);
        system("node _put_.js");
    }
    else
    {
        std::cerr << "CANNOT OPEN FILE!" << std::endl;
    }
}

void serverInit(Board board)
{
    system("powershell.exe -command \"& {Start-Process -WindowStyle min node ../UI/server.js}\"");
    wait(200);
    setBoardCode(board);
}
