const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("voxcpmShell", {
  onStatus(callback) {
    ipcRenderer.on("status", (_event, payload) => callback(payload));
  },
  getShellState() {
    return ipcRenderer.invoke("get-shell-state");
  },
});
