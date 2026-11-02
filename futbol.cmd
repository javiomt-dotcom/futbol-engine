@echo off
rem Abre la terminal con el Python de Anaconda de esta PC.
setlocal
chcp 65001 >nul
set "CONDA=%USERPROFILE%\anaconda3"
set "PATH=%CONDA%;%CONDA%\Library\mingw-w64\bin;%CONDA%\Library\usr\bin;%CONDA%\Library\bin;%CONDA%\Scripts;%PATH%"
cd /d "%~dp0"
"%CONDA%\python.exe" terminal.py
if "%~1"=="" pause
