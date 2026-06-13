Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = "F:\.VoxCPM\VoxCPM"
venvPythonw = projectDir & "\.venv\Scripts\pythonw.exe"
venvPython = projectDir & "\.venv\Scripts\python.exe"
scriptPath = projectDir & "\voxcpm_dev_shell.py"

If fso.FileExists(venvPythonw) Then
    pythonExe = venvPythonw
ElseIf fso.FileExists(venvPython) Then
    pythonExe = venvPython
Else
    pythonExe = "pythonw.exe"
End If

shell.CurrentDirectory = projectDir
shell.Run """" & pythonExe & """ """ & scriptPath & """", 0, False
