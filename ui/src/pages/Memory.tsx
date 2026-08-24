import { useEffect, useState } from "react";
import { api, type Scope } from "../api/client";

export function Memory() {
  const [scope, setScope] = useState<Scope | undefined>(undefined);
  const [items, setItems] = useState<Array<Record<string, unknown>>>([]);
  const load = () => api.memories(scope).then(setItems).catch(() => setItems([]));
  useEffect(() => { void load(); }, [scope]);
  const edit = async (item: Record<string, unknown>) => {
    const value = window.prompt("Update memory", String(item.value ?? ""));
    if (value && value !== item.value) {
      await api.updateMemory(String(item.key), { value, scope: item.scope as Scope });
      await load();
    }
  };
  const forget = async (item: Record<string, unknown>) => {
    if (window.confirm("Forget this memory?")) {
      await api.forgetMemory(String(item.key), item.scope as Scope);
      await load();
    }
  };
  const move = async (item: Record<string, unknown>) => {
    const target = window.prompt("Move to global, project, or session", "project") as Scope | null;
    if (target && ["global", "project", "session"].includes(target)) {
      await api.moveMemory(String(item.key), target, item.scope as Scope);
      await load();
    }
  };
  return <section className="standard-page"><div className="page-heading"><div><span className="eyebrow">MEMORY / EXPLICIT STATE</span><h2>Remembered,<br /><em>with boundaries.</em></h2></div><button className="primary-button">+ Store memory</button></div><div className="filter-row">{[undefined, "global", "project", "session"].map((value) => <button key={value ?? "all"} className={scope === value ? "filter active" : "filter"} onClick={() => setScope(value as Scope | undefined)}>{value ?? "all"}</button>)}</div><div className="memory-table"><div className="table-head"><span>KEY</span><span>VALUE</span><span>SCOPE</span><span>ACTIONS</span></div>{items.length ? items.map((item) => <div className="table-row" key={String(item.key)}><code>{String(item.key)}</code><span>{String(item.value)}</span><span className={`scope-tag ${String(item.scope)}`}>{String(item.scope)}</span><span className="row-actions"><button onClick={() => void edit(item)}>EDIT</button><button onClick={() => void move(item)}>MOVE</button><button onClick={() => void forget(item)}>FORGET</button></span></div>) : <div className="empty-state">No memories in this scope yet.</div>}</div></section>;
}
