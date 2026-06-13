/// <reference types="vite/client" />

type ShellStatus = {
  state: "starting" | "ready" | "failed" | "exited" | string;
  message: string;
  detail: string;
};

type ShellState = {
  appMode: "app-shell" | "legacy-webui-dev" | string;
  backendUrl: string;
  mainPort: number;
  projectDir: string;
  outLogPath: string;
  errLogPath: string;
  status: ShellStatus;
};

interface Window {
  voxcpmShell?: {
    onStatus(callback: (payload: ShellStatus) => void): void;
    getShellState(): Promise<ShellState>;
  };
}
