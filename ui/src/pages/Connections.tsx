import { useEffect, useState } from "react";
import { api } from "../api/client";

export function Connections() {
  const [items, setItems] = useState<Array<Record<string, unknown>>>([]);
  useEffect(() => { void api.integrations().then(setItems).catch(() => setItems([])); }, []);
  return <section className="standard-page"><div className="page-heading"><div><span className="eyebrow">CONNECTIONS / AGENT SURFACES</span><h2>One service.<br /><em>Many clients.</em></h2></div></div><div className="connection-grid">{items.map((item) => <article className="connection-card" key={String(item.name)}><div className="connection-icon">{String(item.name).slice(0, 1)}</div><div><h3>{String(item.name)}</h3><p>{String(item.location ?? item.details ?? "Model-neutral interface")}</p></div><span className="connection-status">{String(item.status)}</span></article>)}</div></section>;
}
