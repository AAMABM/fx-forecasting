@echo off
:: Change directory to where the project is located
cd /d "C:\Users\Dell\OneDrive\Desktop\project"

:: Get the current date for the log file name
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "datestamp=%YYYY%-%MM%-%DD%"

:: Make sure Docker is running and the database is up
docker compose up -d db

:: Run the Python pipeline and save the output to a log file
echo ======================================== >> logs\pipeline_%datestamp%.log
echo Run started at %time% >> logs\pipeline_%datestamp%.log
echo ======================================== >> logs\pipeline_%datestamp%.log
python run.py >> logs\pipeline_%datestamp%.log 2>&1

echo Pipeline complete!