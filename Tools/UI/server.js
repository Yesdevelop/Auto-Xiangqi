const http = require("http")
const fs = require("fs")

let boardCode = "null"
let computerMove = "null"
let getBoardCode = () => boardCode

let file = fs.openSync("./_move_.txt", "w+")
fs.writeFileSync(file, "____")
fs.closeSync(file)

try {
    http.createServer((request, response) => {
        const { method, url } = request
        response.setHeader("Access-Control-Allow-Origin", "*")
        response.setHeader("Access-Control-Allow-Methods", "GET, DELETE, PATCH, OPTIONS")
        response.setHeader("Access-Control-Allow-Headers", "Content-Type")

        if (method === "GET" && url === "/boardcode") { // 界面端获取当前棋盘局势图
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end(getBoardCode() + "\n")
        }
        else if (method === "PUT" && url.match("boardcode")) { // 人机方面做出决策更改服务器棋盘局势图
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end("successful\n")
            boardCode = request.url.split("=")[1]
        }
        else if (method === "PUT" && url.match("move")) { // 获取玩家着法
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end("successful\n")
            computerMove = request.url.split("=")[1]
        }
        else if (method == "GET" && url.match("move")) { // 执行玩家着法
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end("successful\n")
            let move = request.url.split("=")[1]
            const fileWrite = () => {
                setTimeout(() => {
                    try {
                        let file = fs.openSync("./_move_.txt", "w+")
                        fs.writeFileSync(file, move)
                        fs.closeSync(file)
                    }
                    catch (e) {
                        fileWrite()
                    }
                }, 50)
            }
            fileWrite()
        }
        else if (method === "GET" && url.match("computer")) { // 界面端获取电脑着法
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end(computerMove + "\n")
        }
        else if (method == "GET" && url.match("undo")) { // 悔棋
            response.writeHead(200, { "Content-Type": "text/plain" })
            response.end("successful\n")
            const fileWrite = () => {
                setTimeout(() => {
                    try {
                        let file = fs.openSync("./_move_.txt", "w+")
                        fs.writeFileSync(file, "undo")
                        fs.closeSync(file)
                    }
                    catch (e) {
                        fileWrite()
                    }
                }, 50)
            }
            fileWrite()
        }
    }).listen(9494)
} catch (e) {
    console.error("创建服务器失败，可能是因为localhost:9494端口被占用。请检查是否同时运行了多个界面端服务器。。")
}
