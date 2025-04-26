const { Builder, By, Key, until } = require('selenium-webdriver')

async function run() {
    console.log('开始执行')

    const driver = new Builder().forBrowser('MicrosoftEdge').build()

    await driver.get('https://play.xiangqi.com/')

    const playComputer = await driver.findElement(By.css('div[title="Play Computer"]'))
    await playComputer.click()

    const bot = await driver.findElement(By.css('.all-bots :nth-child(9)'))
    await bot.click()

    const playButton = await driver.findElement(By.css('.button-wrapper button:nth-child(1)'))
    await playButton.click()

    console.log('关闭浏览器')
    await driver.quit()
}

run()
