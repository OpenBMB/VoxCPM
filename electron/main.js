const { app, BrowserWindow, ipcMain, shell } = require("electron");
const { spawn, execFile } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const projectDir = path.resolve(__dirname, "..");
const mainPort = 8808;
const mainUrl = `http://127.0.0.1:${mainPort}`;
const outLogPath = path.join(projectDir, "voxcpm_webui.out.log");
const errLogPath = path.join(projectDir, "voxcpm_webui.err.log");
const shouldStartLegacyWebUI = process.env.VOXCPM_START_LEGACY_GRADIO === "1";

let mainWindow = null;
let backendProcess = null;
let isQuitting = false;
let lastStatus = { state: "starting", message: "Starting VoxCPM AppShell", detail: "" };

function sendStatus(state, message, detail = "") {
  lastStatus = { state, message, detail };
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send("status", lastStatus);
}

function pythonPath() {
  const venvPython = path.join(projectDir, ".venv", "Scripts", "python.exe");
  return fs.existsSync(venvPython) ? venvPython : "python";
}

function truncateLogs() {
  fs.writeFileSync(outLogPath, "", "utf8");
  fs.writeFileSync(errLogPath, "", "utf8");
}

function isPortReady() {
  return new Promise((resolve) => {
    const request = http.get(mainUrl, (response) => {
      response.resume();
      resolve(response.statusCode >= 200 && response.statusCode < 500);
    });
    request.on("error", () => resolve(false));
    request.setTimeout(1200, () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(timeoutMs = 180000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await isPortReady()) {
      return true;
    }
    if (backendProcess && backendProcess.exitCode !== null) {
      return false;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return false;
}

function startBackend() {
  if (backendProcess) {
    return;
  }

  truncateLogs();
  const py = pythonPath();
  const args = ["run_with_local_ffmpeg.py", "app.py", "--port", String(mainPort), "--device", "cuda"];
  const outLog = fs.openSync(outLogPath, "a");
  const errLog = fs.openSync(errLogPath, "a");

  sendStatus("starting", "Starting VoxCPM backend", `${py} ${args.join(" ")}`);
  backendProcess = spawn(py, args, {
    cwd: projectDir,
    windowsHide: true,
    stdio: ["ignore", outLog, errLog],
  });
  fs.closeSync(outLog);
  fs.closeSync(errLog);

  backendProcess.on("exit", (code, signal) => {
    if (!isQuitting) {
      sendStatus("exited", "Backend exited", `code=${code ?? ""} signal=${signal ?? ""}`);
    }
  });
  backendProcess.on("error", (error) => {
    sendStatus("failed", "Failed to launch backend", error.stack || String(error));
  });
}

function cleanupResidualBackends() {
  return new Promise((resolve) => {
    const escapedProject = projectDir.replace(/'/g, "''");
    const command = [
      "$rows = Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\"",
      `| Where-Object { $_.CommandLine -like '*${escapedProject.replace(/\\/g, "\\\\")}*app.py --port 8808*' -or $_.CommandLine -like '*app.py --port 8808 --device cuda*' };`,
      "foreach ($row in $rows) { Stop-Process -Id $row.ProcessId -Force -ErrorAction SilentlyContinue }"
    ].join(" ");

    execFile(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
      { cwd: projectDir, windowsHide: true },
      () => resolve()
    );
  });
}

async function stopBackend() {
  isQuitting = true;

  if (backendProcess && backendProcess.exitCode === null) {
    backendProcess.kill();
  }
  backendProcess = null;

  if (shouldStartLegacyWebUI) {
    await cleanupResidualBackends();
  }
}

async function bootWebUI() {
  startBackend();
  sendStatus("starting", "Starting VoxCPM backend", mainUrl);

  const ready = await waitForBackend();
  if (!ready) {
    sendStatus("failed", "Failed to start WebUI", `Check logs:\n${outLogPath}\n${errLogPath}`);
    return;
  }

  sendStatus("ready", "VoxCPM WebUI is ready", mainUrl);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1120,
    minHeight: 760,
    title: "VoxCPM",
    backgroundColor: "#101214",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.removeMenu();
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  const rendererUrl = process.env.VITE_DEV_SERVER_URL;
  if (rendererUrl) {
    mainWindow.loadURL(rendererUrl);
  } else {
    mainWindow.loadFile(path.join(projectDir, "dist", "renderer", "index.html"));
  }
}

ipcMain.handle("get-shell-state", () => ({
  appMode: shouldStartLegacyWebUI ? "legacy-webui-dev" : "app-shell",
  backendUrl: mainUrl,
  mainPort,
  projectDir,
  outLogPath,
  errLogPath,
  status: lastStatus,
}));

app.whenReady().then(() => {
  createWindow();
  if (shouldStartLegacyWebUI) {
    bootWebUI().catch((error) => {
      sendStatus("failed", "Startup error", error.stack || String(error));
    });
    return;
  }
  sendStatus(
    "ready",
    "VoxCPM AppShell is ready",
    "App mode keeps Gradio on the original launcher route instead of embedding it."
  );
});

app.on("before-quit", async (event) => {
  if (isQuitting) {
    return;
  }
  event.preventDefault();
  await stopBackend();
  app.quit();
});

app.on("window-all-closed", () => {
  app.quit();
});
