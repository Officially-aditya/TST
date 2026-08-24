import { useState } from "react";
import type { ContextItem, ContextPack, Scope } from "../api/client";

const scopeLabels: Record<Scope, string> = {
  global: "Shared memory",
  project: "Project memory",
  session: "Current session",
};

function sourceLabel(item: ContextItem): string {
  return item.source === "tree" ? "Relevant code" : "Memory note";
}

function relevanceLabel(score: number): string {
  if (score >= 0.8) return "Strong match";
  if (score >= 0.6) return "Good match";
  return "Possible match";
}

function itemTitle(item: ContextItem): string {
  if (item.source === "tree") return item.symbol ?? item.file ?? "Code reference";
  if (item.scope === "global") return "Shared note";
  if (item.scope === "session") return "Session note";
  return "Project note";
}

export function Context({
  pack,
  query,
  setQuery,
  retrieve,
}: {
  pack: ContextPack | null;
  query: string;
  setQuery: (value: string) => void;
  retrieve: (query?: string) => void;
}) {
  const [focused, setFocused] = useState<number | null>(null);

  return (
    <section className="context-page">
      <div className="context-intro">
        <span className="eyebrow">CONTEXT INSPECTOR / RETRIEVAL PREVIEW</span>
        <h2>
          What would TST
          <br />
          <em>put in the room?</em>
        </h2>
        <p>Readable notes from memory and relevant code, ready for the agent to use.</p>
      </div>
      <form
        className="query-box"
        onSubmit={(event) => {
          event.preventDefault();
          retrieve();
        }}
      >
        <span className="prompt-sign">›</span>
        <input value={query} onChange={(event) => setQuery(event.target.value)} aria-label="Context query" />
        <button type="submit">
          RETRIEVE <span>↗</span>
        </button>
      </form>
      {pack ? (
        <div className="context-results">
          <div className="result-header">
            <span>{pack.items.length} RELEVANT NOTES</span>
            <span>CONTEXT READY</span>
          </div>
          <div className="context-notice">
            TST assembled this background from the current project. It can be incomplete or out of date, so treat it as reference material.
          </div>
          <div className="result-list">
            {pack.items.map((item, index) => (
              <article
                className={focused === index ? "context-item focused" : "context-item"}
                key={`${item.key ?? item.symbol ?? item.file ?? item.source}-${index}`}
                onClick={() => setFocused(index)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") setFocused(index);
                }}
                role="button"
                tabIndex={0}
              >
                <div className="item-score">
                  <span className="relevance-mark" />
                  <span>{relevanceLabel(item.score)}</span>
                </div>
                <div className="item-body">
                  <div className="item-meta">
                    <span className={`scope-tag ${item.scope}`}>{scopeLabels[item.scope]}</span>
                    <span>{sourceLabel(item)}</span>
                  </div>
                  <h3>{itemTitle(item)}</h3>
                  <p>{item.content}</p>
                  {item.file && <small>{item.file}</small>}
                </div>
                <div className="item-arrow">↗</div>
              </article>
            ))}
          </div>
        </div>
      ) : (
        <div className="empty-state context-empty">Run a query to inspect the assembled context.</div>
      )}
    </section>
  );
}
