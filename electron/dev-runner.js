const { spawn } = require("child_process");
const http = require("http");
const electronPath = require("electron");

const rendererUrl = "http://127.0.0.1:17888";
const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
const commandShell = process.env.ComSpec || "cmd.exe";

function pipeOutput(child) {
  child.stdout?.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr?.on("data", (chunk) => process.stderr.write(chunk));
}

function waitForRenderer(timeoutMs = 120000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(rendererUrl, (res) => {
        res.resume();
        resolve();
      });
      req.on("error", () => {
        if (Date.now() - startedAt > timeoutMs) {
          reject(new Error(`Renderer did not start at ${rendererUrl}`));
          return;
        }
        setTimeout(tick, 500);
      });
      req.setTimeout(1000, () => {
        req.destroy();
        setTimeout(tick, 500);
      });
    };
    tick();
  });
}

const viteCommand = process.platform === "win32" ? commandShell : npmCmd;
const viteArgs =
  process.platform === "win32" ? ["/d", "/s", "/c", npmCmd, "run", "renderer:dev"] : ["run", "renderer:dev"];

const vite = spawn(viteCommand, viteArgs, {
  stdio: ["ignore", "pipe", "pipe"],
  shell: false,
});
pipeOutput(vite);

vite.on("exit", (code) => {
  if (code !== 0) {
    process.exit(code ?? 1);
  }
});

waitForRenderer()
  .then(() => {
    const electron = spawn(electronPath, ["."], {
      stdio: ["ignore", "pipe", "pipe"],
      shell: false,
      env: {
        ...process.env,
        VITE_DEV_SERVER_URL: rendererUrl,
      },
    });
    pipeOutput(electron);

    electron.on("exit", (code) => {
      vite.kill();
      process.exit(code ?? 0);
    });
  })
  .catch((error) => {
    console.error(error);
    vite.kill();
    process.exit(1);
  });
