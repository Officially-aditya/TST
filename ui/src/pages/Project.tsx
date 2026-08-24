import type { Status } from "../api/client";

export function Project({ status, onContext }: { status: Status; onContext: () => void }) {
  const files = status.memory_counts.project + status.memory_counts.global;
  return <section className="page-grid">
    <div className="hero-panel"><span className="eyebrow">PROJECT OVERVIEW</span><h2>{status.project.name}<br /><em>is in focus.</em></h2><p className="lede">A quiet control surface for the context your agents can actually see.</p><button className="primary-button" onClick={onContext}>Inspect context <span>→</span></button></div>
    <div className="metrics-panel"><div className="panel-title"><span>RUNTIME</span><span className="live-label">● LIVE</span></div><div className="metric-grid"><Metric label="Index" value={status.kernel.project.running ? "CURRENT" : "IDLE"} /><Metric label="Kernel" value={status.kernel.project.running ? "HEALTHY" : "OFFLINE"} /><Metric label="Memory" value={String(files).padStart(2, "0")} /><Metric label="Session" value={String(status.memory_counts.session).padStart(2, "0")} /></div></div>
    <div className="wide-panel"><div className="panel-title"><span>MEMORY AT A GLANCE</span><span className="muted">SCOPED / EXPLICIT</span></div><div className="scope-strip"><ScopeCard label="Global" value={status.memory_counts.global} color="gold" /><ScopeCard label="Project" value={status.memory_counts.project} color="violet" /><ScopeCard label="Session" value={status.memory_counts.session} color="blue" /></div></div>
    <div className="note-panel"><span className="eyebrow">DESIGN PRINCIPLE</span><p>Nothing moves from project to global scope without a deliberate action.</p><span className="note-mark">TST / 03</span></div>
  </section>;
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div>; }
function ScopeCard({ label, value, color }: { label: string; value: number; color: string }) { return <div className={`scope-card ${color}`}><span>{label}</span><strong>{value.toString().padStart(2, "0")}</strong><small>items</small></div>; }
