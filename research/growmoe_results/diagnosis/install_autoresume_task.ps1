# OPTIONAL. Makes Windows resume the diagnosis study by itself a minute after you log on (for example after a power cut and reboot).
# Not installed by Claude: run it yourself only if you want it.   powershell -ExecutionPolicy Bypass -File install_autoresume_task.ps1
# To remove it later:   schtasks /Delete /TN "GrowMoE-diagnosis-resume" /F
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "C:\Users\Asta\AppData\Local\Programs\Python\Python312\python.exe"
$action = New-ScheduledTaskAction -Execute $py -Argument "`"$here\jobqueue.py`" run --detach" -WorkingDirectory $here
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT1M"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "GrowMoE-diagnosis-resume" -Action $action -Trigger $trigger -Settings $settings -Description "Resumes the GrowMoE diagnosis experiments (idempotent) after logon" -Force
Write-Host "Installed. It will run 'jobqueue.py run --detach' one minute after you log on."
