@echo off
for /L %%i in (1,1,12) do (
    start "" "E:\Projects_chess\Chess98\x64\Release\Chess98.exe"
    timeout /t 1 >nul
)