@echo off
SETLOCAL EnableDelayedExpansion

rem Random port
set /a "port=(%RANDOM% %% 1000) + 6006"

rem Find latest directory in ./train/
for /f "tokens=* delims=" %%a in ('dir /b /ad /o-d "./train/"') do (
    set "logdir=.\train\%%a"
    goto :next
)
:next

echo Starting tensorboard: !logdir!
start http://localhost:!port!
tensorboard --logdir=!logdir! --port=!port!