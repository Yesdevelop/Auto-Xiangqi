const http = require('http');
const fs = require('fs')

let boardCode = 'null'
let computerMove = 'null'
let getBoardCode = () => boardCode

let file = fs.openSync('./_move_.txt', 'w+')
fs.writeFileSync(file, '____')
fs.closeSync(file)

http.createServer((request, response) => {
    const { method, url } = request
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Methods', 'GET, DELETE, PATCH, OPTIONS');
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (method === 'GET' && url === '/boardcode') { // 界面端获取当前棋盘局势图
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end(getBoardCode() + '\n')
    }
    else if (method === 'PUT' && url.match('boardcode')) { // 人机方面做出决策更改服务器棋盘局势图
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end('successful\n')
        boardCode = request.url.split('=')[1];
    }
    else if (method === 'PUT' && url.match('move')) {
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end('successful\n')
        computerMove = request.url.split('=')[1];
    }
    else if (method == 'GET' && url.match('move')) { // 玩家着法
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end('successful\n')
        let move = request.url.split('=')[1];
        let file = fs.openSync('./_move_.txt', 'w+')
        fs.writeFileSync(file, move)
        fs.closeSync(file)
    }
    else if (method === 'GET' && url.match('computer')) {
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end(computerMove + '\n')
    }
    else if (method == 'GET' && url.match('undo')) {
        response.writeHead(200, { 'Content-Type': 'text/plain' });
        response.end('successful\n')
        let file = fs.openSync('./_move_.txt', 'w+')
        fs.writeFileSync(file, 'undo')
        fs.closeSync(file)
    }
}).listen(9494)

console.log('Server running at http://127.0.0.1:9494/')
