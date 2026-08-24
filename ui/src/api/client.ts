export type Scope = "global" | "project" | "session";

export type ContextItem = {
  source: "memory" | "tree";
  scope: Scope;
  key?: string;
  content: string;
  score: number;
  reason: string;
  symbol?: string;
  file?: string;
};

export type ContextPack = {
  query: string;
  project: string;
  items: ContextItem[];
  estimated_tokens: number;
};

export type Status = {
  project: { id: string; name: string; root: string };
  session_id: string;
  kernel: { global: { running: boolean }; project: { running: boolean } };
  memory_counts: Record<Scope, number>;
  healthy: boolean;
};

const json = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const session = window.sessionStorage.getItem("tst-session");
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(session ? { "X-TST-Session": session } : {}),
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) throw new Error(await response.text());
  const result = (await response.json()) as T & { ui_session_token?: string };
  if (result.ui_session_token) window.sessionStorage.setItem("tst-session", result.ui_session_token);
  return result;
};

export const api = {
  status: () => json<Status>("/api/v1/status"),
  context: (query: string, budget = 2000) =>
    json<ContextPack>("/api/v1/context/preview", {
      method: "POST",
      body: JSON.stringify({ query, budget }),
    }),
  memories: (scope?: Scope) =>
    json<Array<Record<string, unknown>>>(`/api/v1/memories${scope ? `?scope=${scope}` : ""}`),
  updateMemory: (key: string, payload: { value: string; scope?: Scope }) =>
    json<Record<string, unknown>>(`/api/v1/memories/${encodeURIComponent(key)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  forgetMemory: (key: string, scope?: Scope) =>
    json<Record<string, unknown>>(`/api/v1/memories/${encodeURIComponent(key)}${scope ? `?scope=${scope}` : ""}`, {
      method: "DELETE",
    }),
  moveMemory: (key: string, target_scope: Scope, source_scope?: Scope) =>
    json<Record<string, unknown>>(`/api/v1/memories/${encodeURIComponent(key)}/move`, {
      method: "POST",
      body: JSON.stringify({ target_scope, source_scope }),
    }),
  tree: (name: string) => json<Array<Record<string, unknown>>>(`/api/v1/tree/find?name=${encodeURIComponent(name)}`),
  integrations: () => json<Array<Record<string, unknown>>>("/api/v1/integrations"),
};
