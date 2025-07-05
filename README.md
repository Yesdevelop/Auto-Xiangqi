# 车九平八

一个借鉴了象眼引擎的中国象棋 AI，正在持续开发中。

## CLion编译

需要在CLion中指定程序的工作路径在main.cpp所在目录

## 使用

1. Node.js 环境(界面要用，很简单，一路next即可)：https://nodejs.org/en/download
2. 编译Chess98目录下的main.cpp，**输出放在main.cpp同级目录下**
3. 将开局库文件BOOK.DAT解压并放到和编译出的exe文件同级的目录下
4. 双击运行exe文件

## 自动测试

先运行一个 chess98 的 exe 实例
cd 到Tools/Auto/，**开启代理（因为selenium会自动从cdn拉取webdriver依赖，cdn似乎是google的）**，运行 `npm test` 指令
之后会自动运行对弈，可以在Tools/Auto/webdriver.js的开头那部分调整对弈人机级别

## 注意事项

g++可能会编译不通过，请在 https://github.com/meganz/mingw-std-threads/releases/tag/1.0.0 中下载thread库
并放在编译器 include path 或者 main.cpp 的同级目录下面
