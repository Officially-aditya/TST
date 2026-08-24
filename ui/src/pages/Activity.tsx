import { useEffect, useState } from "react";

type Event = { timestamp: number; actor: string; operation: string; scope?: string; duration_ms: number; metadata: Record<string, unknown> };
export function Activity() {
  const [events, setEvents] = useState<Event[]>([]);
  useEffect(() => { void fetch("/api/v1/events/stream").then((response) => response.text()).then((text) => setEvents(text.split("\n\n").filter(Boolean).map((line) => JSON.parse(line.replace(/^data: /, ""))))).catch(() => setEvents([])); }, []);
  return <section className="standard-page"><div className="page-heading"><div><span className="eyebrow">ACTIVITY / LOCAL EVENT STREAM</span><h2>See the<br /><em>control plane move.</em></h2></div><span className="live-pill">● LIVE / LAST 500</span></div><div className="activity-list">{events.length ? [...events].reverse().map((event, index) => <div className="activity-row" key={`${event.timestamp}-${index}`}><time>{new Date(event.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><strong>{event.actor}</strong><span>{event.operation.replaceAll(".", " / ")}</span><span className={`scope-tag ${event.scope ?? "project"}`}>{event.scope ?? "local"}</span><small>{event.duration_ms.toFixed(1)} ms</small></div>) : <div className="empty-state">Activity appears here as TST retrieves, indexes, and changes memory.</div>}</div></section>;
}
