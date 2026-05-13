import axios, { type AxiosInstance, type InternalAxiosRequestConfig } from "axios";

const baseURL =
  (typeof window !== "undefined" && (window as unknown as { __API_BASE__?: string }).__API_BASE__) ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000";

export const TOKEN_STORAGE_KEY = "ofa_admin_token";

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

export const api: AxiosInstance = axios.create({ baseURL });

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getStoredToken();
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      setStoredToken(null);
      if (typeof window !== "undefined" && window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  },
);

export interface ApiKeySummary {
  id: string;
  key_prefix: string;
  label: string;
  allowed_symbols: string[];
  created_at: string;
  expires_at: string | null;
  is_active: boolean;
  last_used_at: string | null;
  usage_count: number;
}

export interface ApiKeyCreateResponse {
  key: ApiKeySummary;
  plaintext_key: string;
}

export interface SystemStatus {
  pipeline_running: boolean;
  last_databento_event: string | null;
  last_compute_per_symbol: Record<string, string | null>;
  last_compute_duration_ms: Record<string, number>;
  rows_per_symbol: Record<string, number>;
  metric_rows_per_symbol: Record<string, number>;
  active_api_keys: number;
}

export interface HealthResponse {
  status: string;
  now: string;
  supported_symbols: string[];
  compute_interval_seconds: number;
  last_compute_per_symbol: Record<string, string | null>;
}

export const Auth = {
  async login(username: string, password: string): Promise<string> {
    const resp = await api.post("/admin/login", { username, password });
    return resp.data.access_token as string;
  },
};

export const ApiKeys = {
  async list(): Promise<ApiKeySummary[]> {
    const resp = await api.get("/admin/api-keys");
    return resp.data;
  },
  async create(payload: {
    label: string;
    allowed_symbols: string[];
    expires_at: string | null;
  }): Promise<ApiKeyCreateResponse> {
    const resp = await api.post("/admin/api-keys", payload);
    return resp.data;
  },
  async update(
    id: string,
    payload: Partial<{
      label: string;
      allowed_symbols: string[];
      expires_at: string | null;
      is_active: boolean;
    }>,
  ): Promise<ApiKeySummary> {
    const resp = await api.patch(`/admin/api-keys/${id}`, payload);
    return resp.data;
  },
  async remove(id: string): Promise<void> {
    await api.delete(`/admin/api-keys/${id}`);
  },
};

export const Status = {
  async health(): Promise<HealthResponse> {
    const resp = await api.get("/health");
    return resp.data;
  },
  async system(): Promise<SystemStatus> {
    const resp = await api.get("/admin/system/status");
    return resp.data;
  },
};

// ── Data Inspector ──────────────────────────────────────────────────────────

export interface InspectorTable {
  name: string;
  rows: number;
  latest_ts: string | null;
  lag_seconds: number | null;
}

export interface InspectorMetricBreakdown {
  metric_type: string;
  rows: number;
  first_seen: string | null;
  last_seen: string | null;
  lag_seconds: number | null;
}

export interface InspectorLatestMetric {
  metric_type: string;
  symbol: string;
  ts: string | null;
  lag_seconds: number | null;
  value: number | null;
  expiration: string | null;
  strike: number | null;
  extra: Record<string, unknown>;
}

export interface InspectorTermStructureRow {
  symbol: string;
  expiration: string | null;
  days_to_expiry: number | null;
  atm_iv: number | null;
  call_25d_iv: number | null;
  put_25d_iv: number | null;
  risk_reversal_25d: number | null;
}

export interface InspectorPinRow {
  symbol: string;
  strike: number | null;
  probability: number | null;
  oi: number | null;
  abs_charm: number | null;
  atm_iv: number | null;
}

export interface InspectorFlowEvent {
  id: string;
  ts: string | null;
  symbol: string;
  expiration: string | null;
  strike: number | null;
  option_type: string;
  event_type: string;
  side: number;
  size: number;
  price: number | null;
  legs: number;
  venues: string[];
}

export interface InspectorAlertEvent {
  id: string;
  ts: string | null;
  rule_id: string | null;
  symbol: string;
  matched: Record<string, unknown>;
  payload: Record<string, unknown>;
}

export interface InspectorChainQuality {
  symbol: string;
  rows_last_hour: number;
  latest_ts: string | null;
  lag_seconds: number | null;
  coverage: {
    bid: number | null;
    ask: number | null;
    last_price: number | null;
    iv: number | null;
    delta: number | null;
    gamma: number | null;
    oi: number | null;
    volume: number | null;
    underlying_price: number | null;
  };
}

export interface InspectorIngesterDiag {
  registry_size?: number;
  book_size?: number;
  schemas_active?: string[];
  schemas_dropped?: string[];
  record_counts?: Record<string, number>;
  sample_record_attrs?: Record<string, Record<string, unknown>>;
  first_record_at?: string | null;
  last_record_at?: string | null;
  connection_attempts?: number;
  last_error?: string | null;
  system_messages?: string[];
  error_messages?: string[];
  parents?: string[];
  supported_symbols?: string[];
  error?: string;
}

export interface InspectorPayload {
  now: string;
  supported_symbols: string[];
  tables: InspectorTable[];
  metric_breakdown: InspectorMetricBreakdown[];
  latest_metrics: InspectorLatestMetric[];
  term_structure: InspectorTermStructureRow[];
  pin_probability: InspectorPinRow[];
  flow_events: InspectorFlowEvent[];
  alerts: {
    rules_total: number;
    rules_enabled: number;
    events: InspectorAlertEvent[];
  };
  chain_quality?: InspectorChainQuality[];
  ingesters?: {
    opra?: InspectorIngesterDiag;
    globex?: InspectorIngesterDiag;
  };
}

export const Inspector = {
  async load(): Promise<InspectorPayload> {
    const resp = await api.get("/admin/inspector");
    return resp.data;
  },
};
