# 车九平八

一个借鉴了象眼引擎的中国象棋 AI，正在持续开发中。

## 使用

> 环境要求：
> - 操作系统：Windows, Linux (MacOS 暂未测试)
> - 依赖：Node.js (要求终端能够使用node命令，以启用UI)
> - 编译器：目前 g++, MSVC 都能过编译

如果想要下棋，请确保 `Chess98/nnuefile.hpp` 中没有 `#define NNUE`，否则会切换到 nnue generate file模式

确保 localhost:9494 没有被占用，这是 UI 的服务器端口。

编译 Chess98/main.cpp 后直接运行输出文件，然后在 `tools/ui/ui.html` 下棋

### 使用自动测试工具

> 环境要求：
> - 操作系统：Windows (Linux, MacOS 暂未测试)
> - 依赖：Node.js, Selenium-Webdriver, Microsoft Edge

一个基于 https://play.xiangqi.com/ 使用 selenium 编写的自动测试工具，

1. 先运行一个 chess98 的 exe 实例
2. cd 到 `tools/Auto/`，若第一次使用则先运行 `npm i selenium-webdriver`，下载 webdriver
3. 终端运行 `node z`，开始自动测试

可以在 `tools/auto/z.js` 的文件头部更改对方人机等级
