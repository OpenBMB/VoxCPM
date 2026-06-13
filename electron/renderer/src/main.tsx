import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AudioWaveform,
  Download,
  FileAudio,
  History,
  Languages,
  Library,
  Mic2,
  MoreHorizontal,
  Play,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Star,
  WandSparkles,
} from "lucide-react";
import "./styles.css";

type PageKey = "design" | "clone" | "ultimate" | "library" | "history" | "settings";
type LanguageCode = "en" | "zh";
type NavItem = {
  key: PageKey;
  labelKey: keyof typeof messages.en;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
};

const messages = {
  en: {
    appTitle: "VoxCPM Desktop",
    navDesign: "Voice Design",
    navClone: "Voice Cloning",
    navUltimate: "Ultimate Cloning",
    navLibrary: "Voice Library",
    navHistory: "History",
    navSettings: "Settings",
    appReady: "AppShell ready",
    starting: "Starting VoxCPM AppShell",
    designDescription: "Create a new voice directly from target text and a control instruction.",
    cloneDescription: "Reuse a saved voice or upload a reference clip for controllable cloning.",
    ultimateDescription: "Use reference audio and transcript text for the full cloning workflow.",
    generate: "Generate",
    saveVoice: "Save Voice",
    language: "Language",
    model: "Model",
    effect: "Effect",
    auto: "Auto",
    english: "English",
    chinese: "Chinese",
    noEffect: "No effect",
    controlInstruction: "Control Instruction",
    targetText: "Target Text",
    controlInstructionText: "A calm, natural voice with clear pacing and a studio-like delivery.",
    targetTextValue:
      "Type the speech content here. App mode will connect this surface to VoxCPM services in the next implementation phase.",
    generationOutput: "Generation Output",
    noGeneration: "No app-mode generation yet",
    appModeNote:
      "Gradio remains available from the original launcher. This AppShell is the new product surface for saved voices, history, settings, and future native generation controls.",
    importVoice: "Import Voice",
    createVoice: "Create Voice",
    download: "Download",
    edit: "Edit",
    more: "More",
    favorite: "Favorite",
    endReached: "You've reached the end",
    runtime: "Runtime",
    backend: "Backend",
    appMode: "App mode",
    port: "Port",
    gradioRoute: "Gradio route",
    gradioRouteValue: "Use start_voxcpm.bat for the original WebUI.",
    localPaths: "Local Paths",
    project: "Project",
    outputLog: "Output log",
    errorLog: "Error log",
    interface: "Interface",
    interfaceLanguage: "Interface language",
    languageHint: "Switches AppShell text only. The legacy Gradio WebUI keeps its own route.",
    studioNarratorNote: "Neutral, clean delivery with controlled warmth.",
    mandarinExplainerNote: "Clear educational voice with medium pace.",
    historyTextOne:
      "If you have never touched AI video workflows, start from text prompts and build scenes iteratively.",
    historyTextTwo:
      "A reusable voice should keep tone stable while each generation can reuse text, prompt, and parameters.",
    historyTextThree: "The reference tone stays consistent while the pacing adapts to the target text.",
    today: "Today",
    daysAgo: "17 days ago",
  },
  zh: {
    appTitle: "VoxCPM 桌面应用",
    navDesign: "声音设计",
    navClone: "声音克隆",
    navUltimate: "极致克隆",
    navLibrary: "音色库",
    navHistory: "历史记录",
    navSettings: "设置",
    appReady: "AppShell 已就绪",
    starting: "正在启动 VoxCPM AppShell",
    designDescription: "通过目标文本和控制提示词直接创建新的声音表达。",
    cloneDescription: "复用已保存音色，或上传参考音频进行可控克隆。",
    ultimateDescription: "使用参考音频和转写文本完成完整克隆流程。",
    generate: "生成",
    saveVoice: "保存音色",
    language: "语言",
    model: "模型",
    effect: "效果",
    auto: "自动",
    english: "英语",
    chinese: "中文",
    noEffect: "无效果",
    controlInstruction: "控制提示词",
    targetText: "目标文本",
    controlInstructionText: "平静自然的声音，语速清晰，带有录音棚质感。",
    targetTextValue: "在这里输入要合成的文本。App 模式会在下一阶段接入 VoxCPM 服务。",
    generationOutput: "生成结果",
    noGeneration: "暂无 App 模式生成结果",
    appModeNote: "Gradio 仍可通过原启动器使用。AppShell 是音色库、历史、设置和未来原生生成控件的新产品界面。",
    importVoice: "导入音色",
    createVoice: "创建音色",
    download: "下载",
    edit: "编辑",
    more: "更多",
    favorite: "收藏",
    endReached: "已经到底了",
    runtime: "运行状态",
    backend: "后端",
    appMode: "应用模式",
    port: "端口",
    gradioRoute: "Gradio 入口",
    gradioRouteValue: "使用 start_voxcpm.bat 启动原始 WebUI。",
    localPaths: "本地路径",
    project: "项目",
    outputLog: "输出日志",
    errorLog: "错误日志",
    interface: "界面",
    interfaceLanguage: "界面语言",
    languageHint: "只切换 AppShell 文案。原 Gradio WebUI 保持独立入口。",
    studioNarratorNote: "中性、干净的讲述声音，带有克制的温度。",
    mandarinExplainerNote: "清晰的教学讲解音色，中等语速。",
    historyTextOne: "如果你还没有接触过 AI 视频工作流，可以从文本提示开始，逐步构建场景。",
    historyTextTwo: "可复用音色应保持语气稳定，同时让每次生成复用文本、提示词和参数。",
    historyTextThree: "参考音色保持一致，语速会根据目标文本自然调整。",
    today: "今天",
    daysAgo: "17 天前",
  },
} as const;

type MessageKey = keyof typeof messages.en;

const navItems: NavItem[] = [
  { key: "design", labelKey: "navDesign", icon: WandSparkles },
  { key: "clone", labelKey: "navClone", icon: Mic2 },
  { key: "ultimate", labelKey: "navUltimate", icon: AudioWaveform },
  { key: "library", labelKey: "navLibrary", icon: Library },
  { key: "history", labelKey: "navHistory", icon: History },
  { key: "settings", labelKey: "navSettings", icon: Settings },
];

function getVoiceCards(t: (key: MessageKey) => string) {
  return [
    {
      name: "Studio Narrator",
      note: t("studioNarratorNote"),
      tags: ["en", "narration"],
      active: true,
    },
    {
      name: "Mandarin Explainer",
      note: t("mandarinExplainerNote"),
      tags: ["zh", "guide"],
      active: false,
    },
  ];
}

function getHistoryRows(t: (key: MessageKey) => string) {
  return [
    {
      voice: "Studio Narrator",
      meta: "en / VoxCPM2 / 0:15",
      text: t("historyTextOne"),
      time: t("today"),
    },
    {
      voice: "Mandarin Explainer",
      meta: "zh / VoxCPM2 / 0:13",
      text: t("historyTextTwo"),
      time: t("daysAgo"),
    },
    {
      voice: "Studio Narrator",
      meta: "en / VoxCPM2 / 0:09",
      text: t("historyTextThree"),
      time: t("daysAgo"),
    },
  ];
}

function App() {
  const [activePage, setActivePage] = useState<PageKey>("design");
  const [language, setLanguage] = useState<LanguageCode>(() => {
    const saved = window.localStorage.getItem("voxcpm-app-language");
    return saved === "zh" || saved === "en" ? saved : "en";
  });
  const [shellState, setShellState] = useState<ShellState | null>(null);
  const [status, setStatus] = useState<ShellStatus>({
    state: "starting",
    message: messages.en.starting,
    detail: "",
  });

  const t = useMemo(() => {
    return (key: MessageKey) => messages[language][key] ?? messages.en[key];
  }, [language]);

  useEffect(() => {
    window.localStorage.setItem("voxcpm-app-language", language);
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);

  useEffect(() => {
    window.voxcpmShell?.getShellState().then((state) => {
      setShellState(state);
      setStatus(state.status);
    });
    window.voxcpmShell?.onStatus((payload) => setStatus(payload));
  }, []);

  const activeNav = useMemo(() => navItems.find((item) => item.key === activePage), [activePage]);
  const appReady = status.state === "ready";

  return (
    <div className="app-shell">
      <aside className="rail" aria-label="Primary">
        <div className="brand-mark">
          <Sparkles size={24} strokeWidth={2.1} />
        </div>
        <nav className="rail-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const label = t(item.labelKey);
            return (
              <button
                key={item.key}
                className={`rail-button ${activePage === item.key ? "active" : ""}`}
                title={label}
                aria-label={label}
                onClick={() => setActivePage(item.key)}
              >
                <Icon size={23} strokeWidth={2.05} />
              </button>
            );
          })}
        </nav>
        <div className="version">dev</div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">{t("appTitle")}</p>
            <h1>{activeNav ? t(activeNav.labelKey) : ""}</h1>
          </div>
          <div className="topbar-actions">
            <LanguageSwitch language={language} setLanguage={setLanguage} />
            <BackendPill status={status} t={t} />
          </div>
        </header>

        {activePage === "design" && (
          <GenerationPage
            mode={t("navDesign")}
            status={status}
            appReady={appReady}
            accent="design"
            description={t("designDescription")}
            language={language}
            t={t}
          />
        )}
        {activePage === "clone" && (
          <GenerationPage
            mode={t("navClone")}
            status={status}
            appReady={appReady}
            accent="clone"
            description={t("cloneDescription")}
            language={language}
            t={t}
          />
        )}
        {activePage === "ultimate" && (
          <GenerationPage
            mode={t("navUltimate")}
            status={status}
            appReady={appReady}
            accent="ultimate"
            description={t("ultimateDescription")}
            language={language}
            t={t}
          />
        )}
        {activePage === "library" && <VoiceLibraryPage t={t} />}
        {activePage === "history" && <HistoryPage t={t} />}
        {activePage === "settings" && (
          <SettingsPage
            status={status}
            shellState={shellState}
            language={language}
            setLanguage={setLanguage}
            t={t}
          />
        )}
      </main>
    </div>
  );
}

function LanguageSwitch({
  language,
  setLanguage,
}: {
  language: LanguageCode;
  setLanguage: (language: LanguageCode) => void;
}) {
  return (
    <div className="language-switch" aria-label="Interface language">
      <Languages size={17} />
      <button
        className={language === "en" ? "active" : ""}
        type="button"
        onClick={() => setLanguage("en")}
      >
        EN
      </button>
      <button
        className={language === "zh" ? "active" : ""}
        type="button"
        onClick={() => setLanguage("zh")}
      >
        中
      </button>
    </div>
  );
}

function BackendPill({ status, t }: { status: ShellStatus; t: (key: MessageKey) => string }) {
  return (
    <div className={`backend-pill ${status.state}`}>
      <span />
      <strong>{status.state === "ready" ? t("appReady") : status.message}</strong>
    </div>
  );
}

function GenerationPage({
  mode,
  status,
  appReady,
  accent,
  description,
  language,
  t,
}: {
  mode: string;
  status: ShellStatus;
  appReady: boolean;
  accent: string;
  description: string;
  language: LanguageCode;
  t: (key: MessageKey) => string;
}) {
  return (
    <section className={`generation-grid ${accent}`}>
      <div className="mode-panel">
        <div className="mode-header">
          <SlidersHorizontal size={20} />
          <span>{mode}</span>
        </div>
        <p className="mode-description">{description}</p>
        <div className="quick-controls">
          <button className="primary-action">
            <Sparkles size={18} />
            {t("generate")}
          </button>
          <button className="ghost-action">
            <Library size={18} />
            {t("saveVoice")}
          </button>
        </div>
        <div className="field-stack">
          <label>
            <span>{t("language")}</span>
            <select defaultValue="auto">
              <option value="auto">{t("auto")}</option>
              <option value="en">{t("english")}</option>
              <option value="zh">{t("chinese")}</option>
            </select>
          </label>
          <label>
            <span>{t("model")}</span>
            <select defaultValue="voxcpm2">
              <option value="voxcpm2">VoxCPM2</option>
            </select>
          </label>
          <label>
            <span>{t("effect")}</span>
            <select defaultValue="none">
              <option value="none">{t("noEffect")}</option>
            </select>
          </label>
        </div>
      </div>

      <div className="native-workbench">
        {!appReady && <LoadingPanel status={status} />}
        {appReady && (
          <>
            <div className="prompt-workspace">
              <label>
                <span>{t("controlInstruction")}</span>
                <textarea key={`control-${language}`} defaultValue={t("controlInstructionText")} />
              </label>
              <label>
                <span>{t("targetText")}</span>
                <textarea key={`target-${language}`} defaultValue={t("targetTextValue")} />
              </label>
            </div>
            <aside className="result-panel">
              <div className="result-header">
                <FileAudio size={20} />
                <h2>{t("generationOutput")}</h2>
              </div>
              <div className="audio-placeholder">
                <Play size={22} />
                <span>{t("noGeneration")}</span>
              </div>
              <p>{t("appModeNote")}</p>
            </aside>
          </>
        )}
      </div>
    </section>
  );
}

function LoadingPanel({ status }: { status: ShellStatus }) {
  return (
    <div className="loading-panel">
      <div className="loading-bar" />
      <h2>{status.message}</h2>
      <pre>{status.detail}</pre>
    </div>
  );
}

function VoiceLibraryPage({ t }: { t: (key: MessageKey) => string }) {
  const voiceCards = getVoiceCards(t);

  return (
    <section className="library-layout">
      <div className="section-actions">
        <button className="ghost-action">
          <Download size={18} />
          {t("importVoice")}
        </button>
        <button className="primary-action">
          <Sparkles size={18} />
          {t("createVoice")}
        </button>
      </div>
      <div className="voice-card-grid">
        {voiceCards.map((voice) => (
          <article className={`voice-card ${voice.active ? "selected" : ""}`} key={voice.name}>
            <div className="voice-card-title">
              <AudioWaveform size={20} />
              <h2>{voice.name}</h2>
            </div>
            <p>{voice.note}</p>
            <div className="tag-row">
              {voice.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <div className="card-actions">
              <button title={t("download")} aria-label={t("download")}>
                <Download size={17} />
              </button>
              <button title={t("edit")} aria-label={t("edit")}>
                <SlidersHorizontal size={17} />
              </button>
              <button title={t("more")} aria-label={t("more")}>
                <MoreHorizontal size={17} />
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function HistoryPage({ t }: { t: (key: MessageKey) => string }) {
  const historyRows = getHistoryRows(t);

  return (
    <section className="history-list">
      {historyRows.map((row) => (
        <article className="history-row" key={`${row.voice}-${row.text}`}>
          <AudioWaveform size={22} className="wave-icon" />
          <div className="history-main">
            <h2>{row.voice}</h2>
            <p>{row.meta}</p>
            <span>{row.time}</span>
          </div>
          <div className="history-text">{row.text}</div>
          <div className="history-actions">
            <button title={t("favorite")} aria-label={t("favorite")}>
              <Star size={19} />
            </button>
            <button title={t("more")} aria-label={t("more")}>
              <MoreHorizontal size={19} />
            </button>
          </div>
        </article>
      ))}
      <div className="end-note">{t("endReached")}</div>
    </section>
  );
}

function SettingsPage({
  status,
  shellState,
  language,
  setLanguage,
  t,
}: {
  status: ShellStatus;
  shellState: ShellState | null;
  language: LanguageCode;
  setLanguage: (language: LanguageCode) => void;
  t: (key: MessageKey) => string;
}) {
  return (
    <section className="settings-grid">
      <div className="settings-panel">
        <h2>{t("runtime")}</h2>
        <dl>
          <dt>{t("backend")}</dt>
          <dd>{status.state}</dd>
          <dt>{t("appMode")}</dt>
          <dd>{shellState?.appMode ?? "app-shell"}</dd>
          <dt>{t("port")}</dt>
          <dd>{shellState?.mainPort ?? 8808}</dd>
          <dt>{t("gradioRoute")}</dt>
          <dd>{t("gradioRouteValue")}</dd>
        </dl>
      </div>
      <div className="settings-panel">
        <h2>{t("interface")}</h2>
        <dl>
          <dt>{t("interfaceLanguage")}</dt>
          <dd>
            <LanguageSwitch language={language} setLanguage={setLanguage} />
          </dd>
          <dt>{t("language")}</dt>
          <dd>{language === "zh" ? t("chinese") : t("english")}</dd>
          <dt>{t("appMode")}</dt>
          <dd>{t("languageHint")}</dd>
        </dl>
      </div>
      <div className="settings-panel">
        <h2>{t("localPaths")}</h2>
        <dl>
          <dt>{t("project")}</dt>
          <dd>{shellState?.projectDir ?? "F:\\.VoxCPM\\VoxCPM"}</dd>
          <dt>{t("outputLog")}</dt>
          <dd>{shellState?.outLogPath ?? "voxcpm_webui.out.log"}</dd>
          <dt>{t("errorLog")}</dt>
          <dd>{shellState?.errLogPath ?? "voxcpm_webui.err.log"}</dd>
        </dl>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
