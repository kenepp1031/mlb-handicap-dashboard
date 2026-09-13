Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\MLB Handicap"
WshShell.Run """C:\MLB Handicap\.venv\Scripts\streamlit.exe"" run ""C:\MLB Handicap\app.py""", 0, False
