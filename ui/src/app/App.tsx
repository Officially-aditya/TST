import { useEffect, useState } from "react";
import { api, type ContextPack, type Scope, type Status } from "../api/client";
import { Activity } from "../pages/Activity";
import { Connections } from "../pages/Connections";
import { Context } from "../pages/Context";
import { Memory } from "../pages/Memory";
import { Project } from "../pages/Project";
import { Tree } from "../pages/Tree";

export type PageName = "project" | "context" | "memory" | "tree" | "activity" | "connections";

const pages: Array<{ id: PageName; label: string; mark: string }> = [
  { id: "project", label: "Project", mark: "01" },
  { id: "context", label: "Context", mark: "02" },
  { id: "memory", label: "Memory", mark: "03" },
  { id: "tree", label: "Tree", mark: "04" },
  { id: "activity", label: "Activity", mark: "05" },
  { id: "connections", label: "Connections", mark: "06" },
];

export function App() {
  const [page, setPage] = useState<PageName>("project");
  const [status, setStatus] = useState<Status | null>(null);
  const [context, setContext] = useState<ContextPack | null>(null);
  const [contextQuery, setContextQuery] = useState("implement authentication middleware");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.status().then(setStatus).catch((reason) => setError(String(reason)));
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => window.clearInterval(timer);
  }, []);

  const retrieve = async (query = contextQuery) => {
    setError(null);
    try {
      setContext(await api.context(query));
      setPage("context");
    } catch (reason) {
      setError(String(reason));
    }
  };

  const renderPage = () => {
    if (!status) return <div className="empty-state">Connecting to the local control plane...</div>;
    switch (page) {
      case "context":
        return <Context pack={context} query={contextQuery} setQuery={setContextQuery} retrieve={retrieve} />;
      case "memory":
        return <Memory />;
      case "tree":
        return <Tree />;
      case "activity":
        return <Activity />;
      case "connections":
        return <Connections />;
      default:
        return <Project status={status} onContext={() => void retrieve()} />;
    }
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-glyph">+</span><span>TST</span><small>CONTROL PLANE</small></div>
        <div className="project-switcher"><span className="eyebrow">ACTIVE PROJECT</span><strong>{status?.project.name ?? "..."}</strong><span className="branch">local / main</span></div>
        <nav className="nav" aria-label="TST sections">
          {pages.map((item) => (
            <button key={item.id} className={page === item.id ? "nav-item active" : "nav-item"} onClick={() => setPage(item.id)}>
              <span className="nav-mark">{item.mark}</span><span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot"><span className={status?.healthy ? "health-dot" : "health-dot bad"} />{status?.healthy ? "LOCAL / HEALTHY" : "ATTENTION REQUIRED"}<span className="version">v0.3</span></div>
      </aside>
      <main className="main">
        <header className="topbar"><div><span className="eyebrow">TST / {page.toUpperCase()}</span><h1>{pages.find((item) => item.id === page)?.label}</h1></div><button className="command-button" onClick={() => void retrieve()}>Run context <kbd>⌘ ↵</kbd></button></header>
        {error && <div className="error-banner">{error}<button onClick={() => setError(null)}>dismiss</button></div>}
        {renderPage()}
      </main>
    </div>
  );
}

export type { Scope };
