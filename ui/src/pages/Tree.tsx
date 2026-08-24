import { useState } from "react";
import { api } from "../api/client";

export function Tree() {
  const [query, setQuery] = useState("AuthService");
  const [nodes, setNodes] = useState<Array<Record<string, unknown>>>([]);
  return <section className="standard-page"><div className="page-heading"><div><span className="eyebrow">TREE / PROJECT GRAPH</span><h2>Follow the<br /><em>relationships.</em></h2></div><form className="tree-search" onSubmit={(event) => { event.preventDefault(); void api.tree(query).then(setNodes); }}><input value={query} onChange={(event) => setQuery(event.target.value)} /><button>FIND</button></form></div><div className="tree-layout"><div className="tree-summary"><span className="eyebrow">SYMBOL QUERY</span><strong>{query}</strong><p>Start with a practical explorer. TST keeps calls, tests, imports, and references bounded.</p></div><div className="tree-results">{nodes.length ? nodes.map((node) => <article className="tree-card" key={String(node.node_id)}><span className="scope-tag project">{String(node.node_type)}</span><h3>{String(node.qualified_name ?? node.name)}</h3><p>{String(node.file_path ?? "project-local")}{node.start_line ? `:${node.start_line}` : ""}</p></article>) : <div className="empty-state">Search for a symbol to inspect its definition and neighbors.</div>}</div></div></section>;
}
