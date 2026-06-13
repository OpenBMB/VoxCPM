import argparse
import os
import socket
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PORT = 8808
LORA_PORT = 7860
MAIN_URL = f"http://127.0.0.1:{MAIN_PORT}"
LORA_URL = f"http://127.0.0.1:{LORA_PORT}"
MAIN_OUT_LOG = PROJECT_DIR / "voxcpm_webui.out.log"
MAIN_ERR_LOG = PROJECT_DIR / "voxcpm_webui.err.log"
LORA_OUT_LOG = PROJECT_DIR / "voxcpm_lora.out.log"
LORA_ERR_LOG = PROJECT_DIR / "voxcpm_lora.err.log"


def python_exe() -> Path:
    venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def pythonw_exe() -> Path:
    venv_pythonw = PROJECT_DIR / ".venv" / "Scripts" / "pythonw.exe"
    if venv_pythonw.exists():
        return venv_pythonw
    executable = Path(sys.executable)
    sibling = executable.with_name("pythonw.exe")
    if sibling.exists():
        return sibling
    return executable


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run_capture(args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_DIR,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
    except Exception as exc:
        return f"$ {' '.join(args)}\nERROR: {exc}\n"

    output = result.stdout.strip()
    error = result.stderr.strip()
    text = f"$ {' '.join(args)}\n"
    if output:
        text += output + "\n"
    if error:
        text += error + "\n"
    text += f"exit_code={result.returncode}\n"
    return text


def process_query(patterns: list[str]) -> list[tuple[int, str]]:
    quoted = " -or ".join([f"$_.CommandLine -like '*{p}*'" for p in patterns])
    command = (
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" "
        f"| Where-Object {{ {quoted} }} "
        "| ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        shell=False,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "\t" not in line:
            continue
        pid, command_line = line.split("\t", 1)
        if pid.strip().isdigit():
            rows.append((int(pid.strip()), command_line.strip()))
    return rows


def stop_processes(patterns: list[str]) -> str:
    rows = process_query(patterns)
    if not rows:
        return "No matching VoxCPM service processes found.\n"

    pids = ",".join(str(pid) for pid, _ in rows)
    command = f"Stop-Process -Id {pids} -Force"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        shell=False,
    )
    text = "Stopped processes:\n"
    for pid, command_line in rows:
        text += f"  {pid}: {command_line}\n"
    if result.stderr.strip():
        text += result.stderr.strip() + "\n"
    return text


def truncate_log(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return f"{path.name} does not exist yet.\n"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:]) + ("\n" if content else "")


def start_service(script: str, args: list[str], out_log: Path, err_log: Path, port: int) -> str:
    if is_port_open(port):
        return f"Port {port} is already open. Service was not started again.\n"
    truncate_log(out_log)
    truncate_log(err_log)
    command = [str(python_exe()), "run_with_local_ffmpeg.py", script, *args]
    out_handle = out_log.open("a", encoding="utf-8")
    err_handle = err_log.open("a", encoding="utf-8")
    try:
        subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=out_handle,
            stderr=err_handle,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    finally:
        out_handle.close()
        err_handle.close()
    return f"Started: {' '.join(command)}\n"


def environment_report() -> str:
    py = python_exe()
    report = [
        f"Project: {PROJECT_DIR}",
        f"Python: {py}",
        f"Main URL: {MAIN_URL}",
        f"LoRA URL: {LORA_URL}",
        f"Main port open: {is_port_open(MAIN_PORT)}",
        f"LoRA port open: {is_port_open(LORA_PORT)}",
        "",
    ]
    report.append(run_capture([str(py), "--version"]))
    report.append(run_capture([str(py), "-m", "pip", "--version"]))
    report.append(
        run_capture(
            [
                str(py),
                "-c",
                "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')",
            ],
            timeout=60,
        )
    )
    report.append(run_capture(["nvidia-smi"], timeout=30))
    return "\n".join(report)


class VoxCPMShell(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VoxCPM Dev Shell")
        self.geometry("980x680")
        self.minsize(820, 560)
        self.configure(bg="#f4f4f2")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.refresh_status()
        self.after(3000, self.auto_refresh)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("TButton", padding=7)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

        outer = ttk.Frame(self, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="VoxCPM Dev Shell", style="Title.TLabel").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Checking status...")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        controls = ttk.LabelFrame(outer, text="Services", padding=12)
        controls.pack(fill=tk.X, pady=(12, 10))

        buttons = [
            ("Start Main CUDA", self.start_main_cuda),
            ("Start Main Auto", self.start_main_auto),
            ("Start Main CPU", self.start_main_cpu),
            ("Stop Main", self.stop_main),
            ("Open Main", lambda: webbrowser.open(MAIN_URL)),
            ("Start LoRA", self.start_lora),
            ("Stop LoRA", self.stop_lora),
            ("Open LoRA", lambda: webbrowser.open(LORA_URL)),
            ("Env Check", self.check_env),
            ("Refresh Logs", self.show_logs),
        ]
        for index, (label, command) in enumerate(buttons):
            ttk.Button(controls, text=label, command=command).grid(
                row=index // 5,
                column=index % 5,
                padx=5,
                pady=5,
                sticky="ew",
            )
        for column in range(5):
            controls.columnconfigure(column, weight=1)

        self.output = scrolledtext.ScrolledText(
            outer,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=24,
            bg="#101214",
            fg="#e8e8e3",
            insertbackground="#e8e8e3",
        )
        self.output.pack(fill=tk.BOTH, expand=True)

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(footer, text="Copy Output", command=self.copy_output).pack(side=tk.LEFT)
        ttk.Button(footer, text="Clear", command=lambda: self.output.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=6)
        ttk.Button(footer, text="Refresh Status", command=self.refresh_status).pack(side=tk.RIGHT)

    def append(self, text: str) -> None:
        self.output.insert(tk.END, text + ("\n" if not text.endswith("\n") else ""))
        self.output.see(tk.END)

    def run_threaded(self, title: str, func) -> None:
        def worker() -> None:
            self.after(0, lambda: self.append(f"\n=== {title} ===\n"))
            try:
                text = func()
            except Exception as exc:
                text = f"ERROR: {exc}\n"
            self.after(0, lambda: self.append(text))
            self.after(0, self.refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_status(self) -> None:
        main = "open" if is_port_open(MAIN_PORT) else "closed"
        lora = "open" if is_port_open(LORA_PORT) else "closed"
        self.status_var.set(f"Main {MAIN_PORT}: {main}   LoRA {LORA_PORT}: {lora}")

    def auto_refresh(self) -> None:
        if self.winfo_exists():
            self.refresh_status()
            self.after(3000, self.auto_refresh)

    def start_main_cuda(self) -> None:
        self.run_threaded(
            "Start Main WebUI CUDA",
            lambda: start_service(
                "app.py",
                ["--port", str(MAIN_PORT), "--device", "cuda"],
                MAIN_OUT_LOG,
                MAIN_ERR_LOG,
                MAIN_PORT,
            ),
        )

    def start_main_auto(self) -> None:
        self.run_threaded(
            "Start Main WebUI Auto",
            lambda: start_service(
                "app.py",
                ["--port", str(MAIN_PORT), "--device", "auto"],
                MAIN_OUT_LOG,
                MAIN_ERR_LOG,
                MAIN_PORT,
            ),
        )

    def start_main_cpu(self) -> None:
        self.run_threaded(
            "Start Main WebUI CPU",
            lambda: start_service(
                "app.py",
                ["--port", str(MAIN_PORT), "--device", "cpu"],
                MAIN_OUT_LOG,
                MAIN_ERR_LOG,
                MAIN_PORT,
            ),
        )

    def start_lora(self) -> None:
        self.run_threaded(
            "Start LoRA WebUI",
            lambda: start_service("lora_ft_webui.py", [], LORA_OUT_LOG, LORA_ERR_LOG, LORA_PORT),
        )

    def stop_main(self) -> None:
        self.run_threaded("Stop Main WebUI", lambda: stop_processes(["app.py --port 8808"]))

    def stop_lora(self) -> None:
        self.run_threaded("Stop LoRA WebUI", lambda: stop_processes(["lora_ft_webui.py"]))

    def check_env(self) -> None:
        self.run_threaded("Environment Check", environment_report)

    def show_logs(self) -> None:
        def logs() -> str:
            return (
                "[Main stdout]\n"
                + tail(MAIN_OUT_LOG)
                + "\n[Main stderr]\n"
                + tail(MAIN_ERR_LOG)
                + "\n[LoRA stdout]\n"
                + tail(LORA_OUT_LOG)
                + "\n[LoRA stderr]\n"
                + tail(LORA_ERR_LOG)
            )

        self.run_threaded("Logs", logs)

    def copy_output(self) -> None:
        text = self.output.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Output copied to clipboard.")

    def on_close(self) -> None:
        self.status_var.set("Stopping backend services...")
        self.update_idletasks()
        stop_processes(["app.py --port 8808", "lora_ft_webui.py"])
        self.destroy()


class LoadingSplash(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VoxCPM")
        self.geometry("420x210")
        self.resizable(False, False)
        self.configure(bg="#111315")
        self.overrideredirect(True)
        self._step = 0
        self._center()
        self._build()
        self.after(80, self._animate)
        self.after(1400, self.destroy)

    def _center(self) -> None:
        self.update_idletasks()
        width = 420
        height = 210
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build(self) -> None:
        container = tk.Frame(self, bg="#111315")
        container.pack(fill=tk.BOTH, expand=True, padx=26, pady=24)
        tk.Label(
            container,
            text="VoxCPM Dev Shell",
            bg="#111315",
            fg="#f3f1ea",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            container,
            text="Preparing launcher and service controls",
            bg="#111315",
            fg="#b8b8ae",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(6, 20))
        self.canvas = tk.Canvas(container, height=18, bg="#202326", highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.message_var = tk.StringVar(value="Checking local environment")
        tk.Label(
            container,
            textvariable=self.message_var,
            bg="#111315",
            fg="#d6d0bd",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(16, 0))

    def _animate(self) -> None:
        self._step += 1
        width = max(self.canvas.winfo_width(), 1)
        self.canvas.delete("bar")
        span = width // 3
        x = (self._step * 18) % (width + span) - span
        self.canvas.create_rectangle(x, 0, x + span, 18, fill="#d7b46a", width=0, tags="bar")
        messages = [
            "Checking local environment",
            "Preparing service controls",
            "Loading application shell",
        ]
        self.message_var.set(messages[(self._step // 8) % len(messages)])
        if self.winfo_exists():
            self.after(80, self._animate)


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM desktop development shell")
    parser.add_argument("--check", action="store_true", help="print environment report and exit")
    args = parser.parse_args()
    if args.check:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(environment_report())
        return 0
    splash = LoadingSplash()
    splash.mainloop()
    app = VoxCPMShell()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
