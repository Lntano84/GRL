@echo off
REM Commit and push in one step, with an unambiguous success/failure report.
REM
REM Why this exists: `git push` writes its progress to stderr, so shells surface a non-zero exit
REM code and an error line EVEN WHEN THE PUSH SUCCEEDED.  That has already produced a false
REM "did you push?" check.  This script decides success by comparing the remote ref to local
REM HEAD, so the verdict reflects reality rather than an exit code.
REM
REM Implemented as .cmd rather than .ps1 because PowerShell execution policy blocks local .ps1
REM files on this machine and working around that with -ExecutionPolicy Bypass is a maintenance
REM burden for no benefit.
REM
REM Usage:
REM   scripts\git_push.cmd "commit subject"
REM   scripts\git_push.cmd path\to\message.txt
REM
REM If the argument names an existing file it is passed to `git commit -F`, otherwise it is used as
REM the subject line.  The message file is never committed.

setlocal enabledelayedexpansion
cd /d "%~dp0.."

set REMOTE=lntano-src
set BRANCH=main

if "%~1"=="" (
  echo ERROR: pass a commit message or a message-file path
  exit /b 2
)

set MSGFILE=
if exist "%~1" set MSGFILE=%~1

git add -A
if defined MSGFILE git reset -q HEAD "%MSGFILE%" 2>nul

for /f "delims=" %%f in ('git diff --cached --name-only') do set STAGED=1

if not defined STAGED (
  echo Nothing staged; no commit made.
) else (
  if defined MSGFILE (
    git -c user.email=grl@local -c user.name="GRL merge" commit -q -F "%MSGFILE%"
  ) else (
    git -c user.email=grl@local -c user.name="GRL merge" commit -q -m "%~1"
  )
  if errorlevel 1 (
    echo ERROR: git commit failed
    exit /b 1
  )
  for /f "delims=" %%h in ('git log --oneline -1') do echo Committed: %%h
)

REM Capture push output so its stderr chatter cannot be mistaken for failure.
git push %REMOTE% HEAD:%BRANCH% > "%TEMP%\gitpush.out" 2>&1

for /f "delims=" %%s in ('git rev-parse HEAD') do set LOCAL=%%s
set REMOTE_SHA=
for /f "tokens=1" %%s in ('git ls-remote %REMOTE% %BRANCH% 2^>nul') do set REMOTE_SHA=%%s

if "%LOCAL%"=="%REMOTE_SHA%" (
  echo PUSHED OK  %REMOTE%/%BRANCH% = %LOCAL:~0,8%
  for /f "delims=" %%d in ('git status --short') do (
    echo WARNING: working tree not clean after push: %%d
  )
  del "%TEMP%\gitpush.out" 2>nul
  exit /b 0
) else (
  echo PUSH FAILED
  echo   local : %LOCAL%
  echo   remote: %REMOTE_SHA%
  echo   push output:
  type "%TEMP%\gitpush.out"
  exit /b 1
)
