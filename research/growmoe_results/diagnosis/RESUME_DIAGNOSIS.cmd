@echo off
rem Resume the GrowMoE diagnosis study after a power cut (or any interruption). Safe to run as often as you like:
rem finished work is skipped, running work is not started twice, interrupted work continues from its checkpoint.
cd /d "%~dp0"
"C:\Users\Asta\AppData\Local\Programs\Python\Python312\python.exe" jobqueue.py run --detach
echo.
"C:\Users\Asta\AppData\Local\Programs\Python\Python312\python.exe" jobqueue.py status
echo.
echo The scheduler keeps working in the background (log: logs\queue_scheduler.log). Run this file again at any time to check or to resume.
pause
