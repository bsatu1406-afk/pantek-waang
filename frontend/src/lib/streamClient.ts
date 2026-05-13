/**
 * Live snapshot stream client.
 *
 * Opens a WebSocket to ``/v1/{symbol}/stream``. Falls back to SSE
 * (``/v1/{symbol}/stream/sse``) if the WebSocket upgrade fails.
 * Reconnects with exponential back-off capped at 30 s.
 *
 * Exposes a React context + ``useLiveSnapshot(symbol)`` hook returning the
 * latest snapshot payload and a connection status flag.
 */

import {
  createContext,
  createElement,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "closed" | "error";

// ── Snapshot payload shape ─────────────────────────────────────────────────
//
// Mirrors ``GET /v1/{symbol}/snapshot`` (and the WS / SSE frame). Fields are
// optional because the backend may omit sections that have no data yet.

export interface GexStrike {
  strike: number;
  net_gex: number;
  call_gex?: number;
  put_gex?: number;
}

export interface GexPayload {
  net_total: number;
  curve: GexStrike[];
  top_positive?: GexStrike[];
  top_negative?: GexStrike[];
  zero_gamma?: number | null;
  underlying_price?: number | null;
}

export interface WallStrike {
  rank: number;
  strike: number;
  value: number;
}

export interface WallsPayload {
  call_wall_oi?: WallStrike[];
  put_wall_oi?: WallStrike[];
  call_wall_volume?: WallStrike[];
  put_wall_volume?: WallStrike[];
}

export interface MaxPainPerExpiry {
  expiration: string;
  strike: number;
  pain: number;
}

export interface MaxPainAggregate {
  strike: number;
  value: number;
}

export interface MaxPainPayload {
  per_expiry: MaxPainPerExpiry[];
  aggregate: MaxPainAggregate | null;
}

export type RegimeLabel = "bullish" | "neutral" | "bearish";

export interface RegimeEntry {
  score: number;
  label: RegimeLabel | string;
  call_wall_total: number;
  put_wall_total: number;
  net_gex: number;
}

export interface RegimePayload {
  oi: RegimeEntry | null;
  vol: RegimeEntry | null;
  label?: RegimeLabel | string;
  score?: number;
}

export interface HiroSeriesPoint {
  ts: string;
  value: number;
}

export interface HiroPayload {
  bucket_size?: string;
  cumulative: number;
  series: HiroSeriesPoint[];
}

export interface FlowEvent {
  id?: string;
  ts: string;
  symbol?: string;
  expiration?: string | null;
  strike?: number | null;
  option_type?: string;
  event_type: "SWEEP" | "BLOCK" | "UOA" | string;
  side: number;
  size?: number;
  premium?: number | null;
  price?: number | null;
  contract_label?: string | null;
  legs?: number;
  venues?: string[];
}

export interface FlowPayload {
  events: FlowEvent[];
  counts?: Record<string, number>;
}

export interface ZeroGammaPayload {
  oi: number | null;
  volume: number | null;
  underlying_price: number | null;
}

export interface SnapshotData {
  gex?: GexPayload;
  gex_volume?: GexPayload;
  zero_gamma?: ZeroGammaPayload;
  max_pain?: MaxPainPayload;
  walls?: WallsPayload;
  iv?: {
    atm_iv: number | null;
    skew_per_expiry: Record<string, number>;
    surface: unknown[];
  };
  regime?: RegimePayload;
  hiro?: HiroPayload;
  flow?: FlowPayload;
}

export interface SnapshotEnvelope {
  symbol: string;
  computed_at: string | null;
  next_update_in_seconds: number;
  data: SnapshotData;
}

// ── Configuration ─────────────────────────────────────────────────────────

const RECONNECT_INITIAL_MS = 500;
const RECONNECT_MAX_MS = 30_000;
const STREAM_API_KEY_STORAGE = "ofa_stream_api_key";

export function getStoredApiKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(STREAM_API_KEY_STORAGE) ?? "";
}

export function setStoredApiKey(value: string): void {
  if (typeof window === "undefined") return;
  if (value) {
    window.localStorage.setItem(STREAM_API_KEY_STORAGE, value);
  } else {
    window.localStorage.removeItem(STREAM_API_KEY_STORAGE);
  }
}

function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const override = (window as unknown as { __API_BASE__?: string }).__API_BASE__;
    if (override) return override;
  }
  return import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
}

function toWsUrl(base: string, path: string, key: string): string {
  const u = new URL(path, base);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  if (key) u.searchParams.set("key", key);
  return u.toString();
}

function toSseUrl(base: string, path: string, key: string): string {
  const u = new URL(path, base);
  if (key) u.searchParams.set("key", key);
  return u.toString();
}

function parseFrame(raw: string): SnapshotEnvelope | null {
  try {
    const obj = JSON.parse(raw) as SnapshotEnvelope;
    if (typeof obj !== "object" || obj === null) return null;
    if (!("data" in obj)) return null;
    return obj;
  } catch {
    return null;
  }
}

// ── Stream connection (WS with SSE fallback) ──────────────────────────────

type StreamHandlers = {
  onFrame: (frame: SnapshotEnvelope) => void;
  onStatus: (status: ConnectionStatus) => void;
};

interface StreamConnection {
  close: () => void;
}

function openStream(
  symbol: string,
  apiKey: string,
  handlers: StreamHandlers,
): StreamConnection {
  let closed = false;
  let ws: WebSocket | null = null;
  let sse: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let backoffMs = RECONNECT_INITIAL_MS;
  let wsHasOpened = false;
  let usingSse = false;
  const base = getApiBaseUrl();

  function scheduleReconnect(): void {
    if (closed) return;
    handlers.onStatus("reconnecting");
    const delay = Math.min(backoffMs, RECONNECT_MAX_MS);
    backoffMs = Math.min(backoffMs * 2, RECONNECT_MAX_MS);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (closed) return;
      connect();
    }, delay);
  }

  function connectSse(): void {
    if (closed) return;
    usingSse = true;
    handlers.onStatus("connecting");
    try {
      const url = toSseUrl(base, `/v1/${encodeURIComponent(symbol)}/stream/sse`, apiKey);
      sse = new EventSource(url, { withCredentials: false });
    } catch {
      scheduleReconnect();
      return;
    }
    sse.onopen = () => {
      if (closed) return;
      backoffMs = RECONNECT_INITIAL_MS;
      handlers.onStatus("open");
    };
    sse.onmessage = (ev: MessageEvent<string>) => {
      const frame = parseFrame(ev.data);
      if (frame) handlers.onFrame(frame);
    };
    sse.onerror = () => {
      if (closed) return;
      // EventSource auto-reconnects internally, but we surface status so the
      // UI shows the indicator change. Close + manual reconnect gives us
      // back-off control parity with the WS path.
      try {
        sse?.close();
      } catch {
        /* ignore */
      }
      sse = null;
      scheduleReconnect();
    };
  }

  function connect(): void {
    if (closed) return;
    if (usingSse) {
      connectSse();
      return;
    }
    handlers.onStatus("connecting");
    wsHasOpened = false;
    try {
      const url = toWsUrl(base, `/v1/${encodeURIComponent(symbol)}/stream`, apiKey);
      ws = new WebSocket(url);
    } catch {
      // Browser rejected the URL entirely — flip to SSE fallback.
      usingSse = true;
      connectSse();
      return;
    }
    ws.onopen = () => {
      if (closed) return;
      wsHasOpened = true;
      backoffMs = RECONNECT_INITIAL_MS;
      handlers.onStatus("open");
    };
    ws.onmessage = (ev: MessageEvent<string | ArrayBuffer | Blob>) => {
      if (typeof ev.data !== "string") return;
      const frame = parseFrame(ev.data);
      if (frame) handlers.onFrame(frame);
    };
    ws.onerror = () => {
      // Errors before the first ``open`` event likely indicate the WS upgrade
      // was rejected (corporate proxy stripping ``Upgrade``, etc.). Flip to
      // SSE permanently for this connection.
      if (!wsHasOpened) {
        usingSse = true;
      }
    };
    ws.onclose = () => {
      ws = null;
      if (closed) {
        handlers.onStatus("closed");
        return;
      }
      scheduleReconnect();
    };
  }

  connect();

  return {
    close: () => {
      closed = true;
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
        ws = null;
      }
      if (sse) {
        try {
          sse.close();
        } catch {
          /* ignore */
        }
        sse = null;
      }
      handlers.onStatus("closed");
    },
  };
}

// ── React context / provider / hook ───────────────────────────────────────

export interface LiveSnapshotValue {
  symbol: string;
  apiKey: string;
  setSymbol: (symbol: string) => void;
  setApiKey: (key: string) => void;
  snapshot: SnapshotEnvelope | null;
  status: ConnectionStatus;
  lastFrameAt: number | null;
}

const LiveSnapshotContext = createContext<LiveSnapshotValue | undefined>(undefined);

export interface LiveSnapshotProviderProps {
  initialSymbol: string;
  children: ReactNode;
}

export function LiveSnapshotProvider({ initialSymbol, children }: LiveSnapshotProviderProps) {
  const [symbol, setSymbolState] = useState<string>(initialSymbol);
  const [apiKey, setApiKeyState] = useState<string>(() => getStoredApiKey());
  const [snapshot, setSnapshot] = useState<SnapshotEnvelope | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("closed");
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null);

  const setSymbol = useCallback((next: string) => {
    setSymbolState(next.trim().toUpperCase());
    setSnapshot(null);
  }, []);

  const setApiKey = useCallback((next: string) => {
    setApiKeyState(next);
    setStoredApiKey(next);
    setSnapshot(null);
  }, []);

  const handlersRef = useRef<StreamHandlers>({
    onFrame: () => {},
    onStatus: () => {},
  });

  handlersRef.current = {
    onFrame: (frame: SnapshotEnvelope) => {
      setSnapshot(frame);
      setLastFrameAt(Date.now());
    },
    onStatus: setStatus,
  };

  useEffect(() => {
    if (!symbol || !apiKey) {
      setStatus("closed");
      return;
    }
    const conn = openStream(symbol, apiKey, {
      onFrame: (f) => handlersRef.current.onFrame(f),
      onStatus: (s) => handlersRef.current.onStatus(s),
    });
    return () => conn.close();
  }, [symbol, apiKey]);

  const value = useMemo<LiveSnapshotValue>(
    () => ({ symbol, apiKey, setSymbol, setApiKey, snapshot, status, lastFrameAt }),
    [symbol, apiKey, setSymbol, setApiKey, snapshot, status, lastFrameAt],
  );

  return createElement(LiveSnapshotContext.Provider, { value }, children);
}

export function useLiveSnapshot(): LiveSnapshotValue {
  const ctx = useContext(LiveSnapshotContext);
  if (!ctx) {
    throw new Error("useLiveSnapshot must be used inside <LiveSnapshotProvider>");
  }
  return ctx;
}

// Exported for tests.
export const __internal = { parseFrame, toWsUrl, toSseUrl };
