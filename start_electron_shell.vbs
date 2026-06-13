Set shell = CreateObject("WScript.Shell")

projectDir = "F:\.VoxCPM\VoxCPM"
shell.CurrentDirectory = projectDir
shell.Run "cmd.exe /c npm.cmd run dev", 0, False
