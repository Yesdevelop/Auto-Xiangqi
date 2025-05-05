const { Builder, By, Key, until, WebDriver } = require('selenium-webdriver')
const edge = require('selenium-webdriver/edge')
const http = require('http')
const { exec } = require('child_process')

let chess98LastMove = "null"
let xiangqiaiLastBoard = [
    undefined,
    [undefined, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [undefined, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [undefined, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [undefined, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    [undefined, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [undefined, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [undefined, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    [undefined, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [undefined, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [undefined, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

async function syncChess98LastMove(driver) {
    http.get('http://localhost:9494/computer', (res) => {
        let data = ''
        res.on('data', (chunk) => {
            data += chunk
        })
        res.on('end', async () => {
            console.log("获取着法成功")
            if (data !== chess98LastMove) {
                chess98LastMove = data
                await doMoveInXiangqiai(driver)
                await driver.sleep(2500)
                await syncXiangqiaiLastMove(driver)
            }
        })
    })
    await driver.sleep(600)
}

async function doMoveInXiangqiai(driver) {
    const x1 = Number(chess98LastMove.charAt(1))
    const y1 = Number(chess98LastMove.charAt(0))
    const x2 = Number(chess98LastMove.charAt(3))
    const y2 = Number(chess98LastMove.charAt(2))
    const _x1 = 11 - (x1 + 1)
    const _y1 = y1 + 1
    const _x2 = 11 - (x2 + 1)
    const _y2 = y2 + 1
    try {
        const actions = driver.actions()

        const start = await driver.findElement(By.css(`#game-grid > div:nth-child(${_x1}) > div:nth-child(${_y1}) > div`))
        const end = await driver.findElement(By.css(`#game-grid > div:nth-child(${_x2}) > div:nth-child(${_y2}) > div`))

        await driver.actions({ bridge: true })
            .move({ origin: start })    // 移动到棋子中心
            .press()                    // 按下鼠标
            .pause(300)                 // 短暂停顿模拟人手操作
            .move({ origin: end })   // 拖到目标位置
            .pause(200)                 // 释放前停顿
            .release()                  // 松开鼠标
            .perform()

        xiangqiaiLastBoard[_x1][_y1] = 0
        xiangqiaiLastBoard[_x2][_y2] = 1
    } catch (error) {
        console.error("步进失败" + error)
        await doMoveInXiangqiai(driver)
    }
}

async function syncXiangqiaiLastMove(driver) {
    let currentBoard = []
    for (let i = 1; i <= 10; i++) {
        for (let j = 1; j <= 9; j++) {
            currentBoard[i] = currentBoard[i] || []
            try {
                const square = await driver.findElement(By.css(`#game-grid > div:nth-child(${i}) > div:nth-child(${j}) > div.square-has-piece`))
                if (square)
                    currentBoard[i][j] = 1
            } catch (error) {
                currentBoard[i][j] = 0
            }
        }
    }
    if (xiangqiaiLastBoard.toString() != currentBoard.toString()) {
        const move = await analyzeChangeToGetMove(driver, xiangqiaiLastBoard, currentBoard)
        if (move.from.x === -1 || move.from.y === -1 || move.to.x === -1 || move.to.y === -1) {
            console.error("move解析失败")
            await doMoveInXiangqiai(driver)
                await driver.sleep(2500)
                await syncXiangqiaiLastMove(driver)
            return
        }
        console.log("走法如下", move)
        xiangqiaiLastBoard = currentBoard
        await updateXiangqiaiChangeToChess98UI(driver, move)
    } else {
        await syncXiangqiaiLastMove(driver)
    }
}

async function analyzeChangeToGetMove(driver, board1, board2) {
    // 获取网页上的棋子
    const elements = await driver.findElements(By.css('.pieces-container > div'))
    let moved = { x: -1, index: -1 }
    let pieces = []
    for (let el of elements) {
        const elChild = await el.findElement(By.css('.pieces-container > div > div > div'))
        const rowNum = 11 - (await el.getAttribute('r'))
        pieces[rowNum] = pieces[rowNum] || []
        pieces[rowNum].push(el)
        if ((await elChild.getAttribute('class')).match('moved-piece')) {
            moved.x = rowNum
            moved.index = pieces[rowNum].length - 1
        }
    }
    // 解析变化
    let moveFrom = { x: -1, y: -1 }
    let moveTo = { x: -1, y: -1 }
    // 如果本来是1的地方变成了0，则定位到了moveFrom
    for (let i = 1; i <= 10; i++) {
        for (let j = 1; j <= 9; j++) {
            if (board1[i][j] == 1 && board2[i][j] == 0) {
                moveFrom.x = i
                moveFrom.y = j
            }
        }
    }
    // 通过moved的index位置来判断y
    let count = 0
    moveTo.x = moved.x
    for (let k in board1[moved.x]) {
        if (board2[moved.x][k] == 1) {
            count++
            if (count == moved.index + 1) {
                moveTo.y = Number(k)
                break
            }
        }
    }
    return { from: moveFrom, to: moveTo }
}

async function updateXiangqiaiChangeToChess98UI(driver, move) {
    console.log("move", move.from)
    const x1 = String(10 - move.from.x)
    const y1 = String(move.from.y - 1)
    const x2 = String(10 - move.to.x)
    const y2 = String(move.to.y - 1)
    console.log(x1, y1, x2, y2)
    const moveString = y1 + x1 + y2 + x2
    console.log("moveString", moveString)
    http.request({
        hostname: '127.0.0.1',
        path: '/move?playermove=' + moveString,
        port: 9494,
        method: 'GET'
    }, (res) => { res.on('error', () => { console.error("motherfucker") }) }).end()
    driver.sleep(1000)
}

async function setupEdgeWithProfile() {
    const options = new edge.Options()

    options.addArguments(
        `--user-data-dir=C:\\Users\\Yeshui\\AppData\\Local\\Microsoft\\Edge\\User Data`,
        `--profile-directory=Default`,
        `--log-level=3`
    )
    const driver = await new Builder()
        .forBrowser('MicrosoftEdge')
        .setEdgeOptions(options)
        .build()

    return driver
}

async function run() {
    exec('taskkill /F /IM msedge.exe', async () => {
        console.log('开始执行')

        const driver = await setupEdgeWithProfile()

        await driver.get('https://play.xiangqi.com/')

        const playComputer = await driver.findElement(By.css('div[title="Play Computer"]'))
        await playComputer.click()

        const bot = await driver.findElement(By.css('.all-bots :nth-child(9)'))
        await bot.click()

        const playButton = await driver.findElement(By.css('.button-wrapper button:nth-child(1)'))
        await playButton.click()

        await driver.sleep(2000)

        while (true) {
            await driver.sleep(1500)
            await syncChess98LastMove(driver)
        }

        console.log('关闭浏览器')
    })
}

run()
