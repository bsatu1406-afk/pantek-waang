import {
  Activity,
  AlertTriangle,
  Database,
  Plug,
  ShieldAlert,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Inspector,
  type InspectorPayload,
  type InspectorLatestMetric,
} from "@/lib/api";
import { cn, formatDateTime, formatRelative } from "@/lib/utils";

const REFRESH_INTERVAL_MS = 30_000;

function lagBadge(lagSeconds: number | null): string {
  if (lagSeconds == null) return "border border-border bg-background/40 text-muted-foreground";
  if (lagSeconds < 120) return "border border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  if (lagSeconds < 600) return "border border-amber-500/40 bg-amber-500/10 text-amber-200";
  return "border border-rose-500/40 bg-rose-500/10 text-rose-200";
}

function lagText(lagSeconds: number | null): string {
  if (lagSeconds == null) return "—";
  if (lagSeconds < 60) return `${Math.round(lagSeconds)}s`;
  if (lagSeconds < 3600) return `${Math.round(lagSeconds / 60)}m`;
  if (lagSeconds < 86400) return `${Math.round(lagSeconds / 3600)}h`;
  return `${Math.round(lagSeconds / 86400)}d`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toFixed(digits);
}

function pickExtra(metric: InspectorLatestMetric, key: string): unknown {
  const ex = metric.extra ?? {};
  return (ex as Record<string, unknown>)[key];
}

function asNumber(v: unknown): number | null {
  if (v == null) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function asArray(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

export function DataInspectorPage() {
  const [payload, setPayload] = useState<InspectorPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await Inspector.load();
        if (cancelled) return;
        setPayload(data);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError((err as Error).message ?? "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const gexVol = useMemo(
    () => payload?.latest_metrics.filter((m) => m.metric_type === "GEX_NET_TOTAL_VOL") ?? [],
    [payload],
  );
  const otherMetrics = useMemo(
    () => payload?.latest_metrics.filter((m) => m.metric_type !== "GEX_NET_TOTAL_VOL") ?? [],
    [payload],
  );

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Data Inspector</h1>
          <p className="text-sm text-muted-foreground">
            Live view of every ingestion table, metric type, flow event and alert. Auto-refresh every 30s.
          </p>
        </div>
        {payload && (
          <div className="text-xs text-muted-foreground">
            Refreshed {formatRelative(payload.now)} • {formatDateTime(payload.now)}
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading && !payload && (
        <div className="rounded-md border border-border bg-background/40 p-6 text-center text-sm text-muted-foreground">
          Loading…
        </div>
      )}

      {/* ── Pipeline health alarm (zero-greek dataset) ──────────────────── */}
      {payload?.chain_quality?.some(
        (c) => (c.coverage.gamma ?? 0) === 0 || (c.coverage.underlying_price ?? 0) === 0,
      ) && (
        <div className="rounded-md border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-100">
          <div className="flex items-center gap-2 font-semibold">
            <ShieldAlert className="h-4 w-4" /> Pipeline diagnostic
          </div>
          <p className="mt-2 leading-relaxed">
            One or more symbols have <span className="font-mono">0%</span> greek or
            <span className="font-mono"> 0%</span> underlying-price coverage. The pipeline
            will silently emit zero-valued metrics in this state. Common causes: the
            Databento OPRA Pillar plan does not include the
            <span className="font-mono"> cmbp-1 </span>
            schema (NBBO updates), or the live ingester is dropping schemas at handshake.
            Check <span className="font-mono">Live ingesters</span> below for
            <span className="font-mono"> schemas_dropped </span>
            and <span className="font-mono">record_counts</span>.
          </p>
        </div>
      )}

      {/* ── Chain data quality (fields populated by upstream feed) ──────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" /> Chain data quality
          </CardTitle>
          <CardDescription>
            Coverage % per field, by symbol. Computed on rows from the last hour
            (or, if none, the latest available). 0% on bid/ask while last_price
            is non-zero usually means cmbp-1 NBBO is not subscribed and the
            pipeline is operating purely off the trade tape.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Rows / 1h</TableHead>
                <TableHead>Latest</TableHead>
                <TableHead>Lag</TableHead>
                <TableHead className="text-right">bid%</TableHead>
                <TableHead className="text-right">ask%</TableHead>
                <TableHead className="text-right">last%</TableHead>
                <TableHead className="text-right">iv%</TableHead>
                <TableHead className="text-right">Δ%</TableHead>
                <TableHead className="text-right">Γ%</TableHead>
                <TableHead className="text-right">OI%</TableHead>
                <TableHead className="text-right">Vol%</TableHead>
                <TableHead className="text-right">spot%</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.chain_quality ?? []).map((c) => (
                <TableRow key={c.symbol}>
                  <TableCell className="font-mono text-xs">{c.symbol}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {c.rows_last_hour.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-xs">
                    {c.latest_ts ? formatRelative(c.latest_ts) : "—"}
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-xs",
                        lagBadge(c.lag_seconds),
                      )}
                    >
                      {lagText(c.lag_seconds)}
                    </span>
                  </TableCell>
                  {(["bid", "ask", "last_price", "iv", "delta", "gamma", "oi", "volume", "underlying_price"] as const).map(
                    (k) => {
                      const v = c.coverage[k];
                      const css =
                        v == null
                          ? "text-muted-foreground"
                          : v >= 95
                            ? "text-emerald-300"
                            : v >= 50
                              ? "text-amber-300"
                              : "text-rose-300";
                      return (
                        <TableCell key={k} className={cn("text-right tabular-nums", css)}>
                          {v == null ? "—" : `${v.toFixed(0)}%`}
                        </TableCell>
                      );
                    },
                  )}
                </TableRow>
              ))}
              {!payload?.chain_quality?.length && (
                <TableRow>
                  <TableCell colSpan={13} className="py-6 text-center text-muted-foreground">
                    No chain data yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── Live ingester diagnostics ───────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plug className="h-4 w-4" /> Live ingesters
          </CardTitle>
          <CardDescription>
            Connection state + schemas the gateway accepted/dropped + cumulative
            record counts by message type. Use this to diagnose subscription
            tier issues — e.g. if record_counts has only{" "}
            <span className="font-mono">TradeMsg / StatMsg</span> and{" "}
            <span className="font-mono">CMBP1Msg = 0</span>, the OPRA plan does
            not include cmbp-1.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {(["opra", "globex"] as const).map((kind) => {
            const diag = payload?.ingesters?.[kind];
            return (
              <div
                key={kind}
                className="rounded-md border border-border bg-background/40 p-4 text-sm"
              >
                <div className="mb-2 flex items-center justify-between">
                  <div className="font-semibold uppercase tracking-wide">{kind}</div>
                  <div className="text-xs text-muted-foreground">
                    {diag?.last_record_at
                      ? `last record ${formatRelative(diag.last_record_at)}`
                      : "no records yet"}
                  </div>
                </div>
                {!diag && (
                  <div className="text-xs text-muted-foreground">No diagnostics.</div>
                )}
                {diag?.error && (
                  <div className="mb-2 text-xs text-rose-300">Error: {diag.error}</div>
                )}
                {diag && !diag.error && (
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <div className="space-y-1 text-xs">
                      <div>
                        <span className="text-muted-foreground">Registry size:</span>{" "}
                        <span className="font-mono">{diag.registry_size ?? "—"}</span>
                      </div>
                      {kind === "globex" && (
                        <div>
                          <span className="text-muted-foreground">Book size:</span>{" "}
                          <span className="font-mono">{diag.book_size ?? "—"}</span>
                        </div>
                      )}
                      <div>
                        <span className="text-muted-foreground">Connection attempts:</span>{" "}
                        <span className="font-mono">{diag.connection_attempts ?? 0}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">First record:</span>{" "}
                        <span className="font-mono">
                          {diag.first_record_at ? formatDateTime(diag.first_record_at) : "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Schemas active:</span>{" "}
                        <span className="font-mono">
                          {(diag.schemas_active ?? []).join(", ") || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Schemas dropped:</span>{" "}
                        <span
                          className={cn(
                            "font-mono",
                            (diag.schemas_dropped ?? []).length > 0
                              ? "text-rose-300"
                              : "text-emerald-300",
                          )}
                        >
                          {(diag.schemas_dropped ?? []).join(", ") || "(none)"}
                        </span>
                      </div>
                      {kind === "globex" && (
                        <div>
                          <span className="text-muted-foreground">Parents:</span>{" "}
                          <span className="font-mono">
                            {(diag.parents ?? []).join(", ") || "—"}
                          </span>
                        </div>
                      )}
                      {diag.last_error && (
                        <div className="break-all">
                          <span className="text-muted-foreground">Last error:</span>{" "}
                          <span className="font-mono text-rose-300">{diag.last_error}</span>
                        </div>
                      )}
                      {(diag.error_messages ?? []).length > 0 && (
                        <div className="break-all">
                          <div className="text-muted-foreground">Recent gateway errors:</div>
                          <ul className="font-mono text-xs text-rose-300 list-disc pl-5">
                            {(diag.error_messages ?? []).slice(-5).map((m, i) => (
                              <li key={i}>{m}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {(diag.system_messages ?? []).length > 0 && (
                        <div className="break-all">
                          <div className="text-muted-foreground">Recent gateway notes:</div>
                          <ul className="font-mono text-xs opacity-80 list-disc pl-5">
                            {(diag.system_messages ?? []).slice(-3).map((m, i) => (
                              <li key={i}>{m}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs mb-1">
                        Cumulative record counts by type:
                      </div>
                      {Object.keys(diag.record_counts ?? {}).length === 0 ? (
                        <div className="text-xs italic text-muted-foreground">
                          No records received yet.
                        </div>
                      ) : (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Type</TableHead>
                              <TableHead className="text-right">Count</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {Object.entries(diag.record_counts ?? {})
                              .sort(([, a], [, b]) => b - a)
                              .map(([k, v]) => (
                                <TableRow key={k}>
                                  <TableCell className="font-mono text-xs">{k}</TableCell>
                                  <TableCell className="text-right tabular-nums">
                                    {v.toLocaleString()}
                                  </TableCell>
                                </TableRow>
                              ))}
                          </TableBody>
                        </Table>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* ── Row counts + freshness ─────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4" /> Ingestion tables
          </CardTitle>
          <CardDescription>
            Row count + age of the latest row per table. Color = freshness (green &lt; 2m, amber &lt; 10m, red ≥ 10m).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {payload?.tables.map((t) => (
              <div
                key={t.name}
                className={cn(
                  "rounded-md p-4",
                  lagBadge(t.lag_seconds),
                )}
              >
                <div className="text-xs uppercase tracking-wide opacity-80">{t.name}</div>
                <div className="mt-1 text-xl font-semibold">{t.rows.toLocaleString()}</div>
                <div className="mt-1 text-xs opacity-90">
                  Latest: {t.latest_ts ? formatRelative(t.latest_ts) : "no rows yet"}
                </div>
                <div className="mt-0.5 text-[11px] opacity-70">
                  {t.latest_ts ? formatDateTime(t.latest_ts) : "—"}
                </div>
                <div className="mt-2 text-[11px] opacity-80">
                  Lag: <span className="font-mono">{lagText(t.lag_seconds)}</span>
                </div>
              </div>
            ))}
            {!payload && (
              <div className="col-span-full text-sm text-muted-foreground">No data.</div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* ── computed_metrics breakdown ─────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-4 w-4" /> computed_metrics by type
          </CardTitle>
          <CardDescription>
            Every metric type the pipeline writes. Lag = time since the last row of that type.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric type</TableHead>
                <TableHead className="text-right">Rows</TableHead>
                <TableHead>First seen</TableHead>
                <TableHead>Last seen</TableHead>
                <TableHead>Lag</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.metric_breakdown ?? []).map((m) => (
                <TableRow key={m.metric_type}>
                  <TableCell className="font-mono text-xs">{m.metric_type}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {m.rows.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-xs">
                    {m.first_seen ? formatDateTime(m.first_seen) : "—"}
                  </TableCell>
                  <TableCell className="text-xs">
                    {m.last_seen ? formatRelative(m.last_seen) : "—"}
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-xs",
                        lagBadge(m.lag_seconds),
                      )}
                    >
                      {lagText(m.lag_seconds)}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
              {!payload?.metric_breakdown?.length && (
                <TableRow>
                  <TableCell colSpan={5} className="py-6 text-center text-muted-foreground">
                    No metrics computed yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── GEX-by-Volume snapshot ────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>GEX-by-Volume — what the indicator renders</CardTitle>
          <CardDescription>
            Top 3 long / short gamma strikes + Zero Gamma per supported symbol.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {gexVol.length === 0 && (
            <div className="text-sm text-muted-foreground">
              No GEX_NET_TOTAL_VOL rows yet — pipeline either hasn’t run or no chain data has arrived.
            </div>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            {gexVol.map((m) => {
              const top_pos = asArray(pickExtra(m, "top_positive"));
              const top_neg = asArray(pickExtra(m, "top_negative"));
              const zero_gamma = asNumber(pickExtra(m, "zero_gamma"));
              const underlying = asNumber(pickExtra(m, "underlying_price"));
              return (
                <div key={m.symbol} className="rounded-md border border-border bg-background/40 p-4">
                  <div className="flex items-baseline justify-between">
                    <div>
                      <div className="font-mono text-sm">{m.symbol}</div>
                      <div className="text-xs text-muted-foreground">
                        as of {formatRelative(m.ts)} ({formatDateTime(m.ts)})
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-muted-foreground">underlying</div>
                      <div className="font-mono text-sm">{formatNumber(underlying)}</div>
                    </div>
                  </div>
                  <div className="mt-3 grid gap-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Net total</span>
                      <span className="font-mono">{formatNumber(m.value)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Zero Gamma</span>
                      <span className="font-mono">{formatNumber(zero_gamma)}</span>
                    </div>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="mb-1 text-emerald-400">Top long-gamma</div>
                      {top_pos.slice(0, 3).map((row, idx) => {
                        const r = row as Record<string, unknown>;
                        return (
                          <div key={idx} className="flex justify-between font-mono">
                            <span>{idx + 1}. {asNumber(r.strike) ?? "?"}</span>
                            <span>{formatNumber(asNumber(r.value))}</span>
                          </div>
                        );
                      })}
                      {top_pos.length === 0 && <span className="opacity-60">—</span>}
                    </div>
                    <div>
                      <div className="mb-1 text-rose-400">Top short-gamma</div>
                      {top_neg.slice(0, 3).map((row, idx) => {
                        const r = row as Record<string, unknown>;
                        return (
                          <div key={idx} className="flex justify-between font-mono">
                            <span>{idx + 1}. {asNumber(r.strike) ?? "?"}</span>
                            <span>{formatNumber(asNumber(r.value))}</span>
                          </div>
                        );
                      })}
                      {top_neg.length === 0 && <span className="opacity-60">—</span>}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* ── Other latest metric snapshots ─────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>Other latest metrics</CardTitle>
          <CardDescription>
            Latest single value of every other metric type per symbol (Vanna, Charm, Max Pain, ATM IV, Move Tracker, Regime, HIRO, Basis, Volume Profile).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead>Strike</TableHead>
                <TableHead>As of</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {otherMetrics.map((m) => (
                <TableRow key={`${m.metric_type}-${m.symbol}-${m.ts}`}>
                  <TableCell className="font-mono text-xs">{m.metric_type}</TableCell>
                  <TableCell className="font-mono text-xs">{m.symbol}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(m.value)}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {m.strike ? formatNumber(m.strike) : "—"}
                  </TableCell>
                  <TableCell className="text-xs">
                    {formatRelative(m.ts)}
                  </TableCell>
                </TableRow>
              ))}
              {otherMetrics.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="py-6 text-center text-muted-foreground">
                    No metrics yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── IV Term Structure ─────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>IV Term Structure</CardTitle>
          <CardDescription>ATM IV, 25Δ Call/Put IV, Risk Reversal — per expiration.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Expiration</TableHead>
                <TableHead className="text-right">DTE</TableHead>
                <TableHead className="text-right">ATM IV</TableHead>
                <TableHead className="text-right">25Δ Call</TableHead>
                <TableHead className="text-right">25Δ Put</TableHead>
                <TableHead className="text-right">RR 25Δ</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.term_structure ?? []).map((row, idx) => (
                <TableRow key={`${row.symbol}-${row.expiration}-${idx}`}>
                  <TableCell className="font-mono">{row.symbol}</TableCell>
                  <TableCell>{row.expiration ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.days_to_expiry ?? "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.atm_iv != null ? `${(row.atm_iv * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.call_25d_iv != null ? `${(row.call_25d_iv * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.put_25d_iv != null ? `${(row.put_25d_iv * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.risk_reversal_25d != null
                      ? `${(row.risk_reversal_25d * 100).toFixed(2)}%`
                      : "—"}
                  </TableCell>
                </TableRow>
              ))}
              {!payload?.term_structure?.length && (
                <TableRow>
                  <TableCell colSpan={7} className="py-6 text-center text-muted-foreground">
                    No term-structure rows yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── Pin Probability ───────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>Pin Probability — top 15</CardTitle>
          <CardDescription>0DTE strike pinning probability (OI × |Charm| × Gaussian kernel).</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Strike</TableHead>
                <TableHead className="text-right">P(pin)</TableHead>
                <TableHead className="text-right">OI</TableHead>
                <TableHead className="text-right">|Charm|</TableHead>
                <TableHead className="text-right">ATM IV</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.pin_probability ?? []).map((row, idx) => (
                <TableRow key={`${row.symbol}-${row.strike}-${idx}`}>
                  <TableCell className="font-mono">{row.symbol}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(row.strike)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.probability != null ? `${(row.probability * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(row.oi, 0)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(row.abs_charm, 4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.atm_iv != null ? `${(row.atm_iv * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                </TableRow>
              ))}
              {!payload?.pin_probability?.length && (
                <TableRow>
                  <TableCell colSpan={6} className="py-6 text-center text-muted-foreground">
                    No pin-probability rows yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── Flow events ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-4 w-4" /> Recent flow events
          </CardTitle>
          <CardDescription>
            Latest 50 SWEEP / BLOCK / UOA events detected from the OPRA trade tape.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Strike</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Price</TableHead>
                <TableHead className="text-right">Legs</TableHead>
                <TableHead>Venues</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.flow_events ?? []).map((e) => (
                <TableRow key={e.id}>
                  <TableCell className="text-xs">{formatRelative(e.ts)}</TableCell>
                  <TableCell className="font-mono">{e.symbol}</TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "rounded px-2 py-0.5 text-xs font-medium",
                        e.event_type === "SWEEP" && "bg-rose-500/10 text-rose-300",
                        e.event_type === "BLOCK" && "bg-sky-500/10 text-sky-300",
                        e.event_type === "UOA" && "bg-amber-500/10 text-amber-300",
                      )}
                    >
                      {e.event_type}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {e.strike}
                    {e.option_type}
                  </TableCell>
                  <TableCell>
                    {e.side > 0 ? "BUY" : e.side < 0 ? "SELL" : "—"}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {e.size.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(e.price)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{e.legs}</TableCell>
                  <TableCell className="text-xs">{e.venues.join(", ") || "—"}</TableCell>
                </TableRow>
              ))}
              {!payload?.flow_events?.length && (
                <TableRow>
                  <TableCell colSpan={9} className="py-6 text-center text-muted-foreground">
                    No flow events yet (will populate when OPRA trade tape is active during US market hours).
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ── Alerts ────────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" /> Alerts
          </CardTitle>
          <CardDescription>
            {(payload?.alerts.rules_total ?? 0)} rules total · {(payload?.alerts.rules_enabled ?? 0)} enabled · last 20 firings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Rule</TableHead>
                <TableHead>Matched</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(payload?.alerts.events ?? []).map((e) => (
                <TableRow key={e.id}>
                  <TableCell className="text-xs">{formatRelative(e.ts)}</TableCell>
                  <TableCell className="font-mono">{e.symbol}</TableCell>
                  <TableCell className="font-mono text-xs">
                    {e.rule_id ? e.rule_id.slice(0, 8) : "—"}
                  </TableCell>
                  <TableCell className="font-mono text-[11px]">
                    {JSON.stringify(e.matched).slice(0, 120)}
                  </TableCell>
                </TableRow>
              ))}
              {!payload?.alerts.events?.length && (
                <TableRow>
                  <TableCell colSpan={4} className="py-6 text-center text-muted-foreground">
                    No alert events yet.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
