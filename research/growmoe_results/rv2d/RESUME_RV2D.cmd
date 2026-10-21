@echo off
rem Resume the RV2-D runs (routing v2 stage 4) after a power cut: finished cells are skipped, checkpointed cells resume.
rem Usage: RESUME_RV2D.cmd dev1|dev2      (confirm is CLOSED since 2026-10-21, see CONFIRM_CLOSED.json: seeds 85-99 are single use and the queue refuses to start them again)
cd /d "%~dp0"
python jobqueue.py run --detach --phase %1
