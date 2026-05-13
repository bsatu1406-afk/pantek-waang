import { useEffect, useState } from "react";
import { Activity, Clock, Database, KeyRound } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Status, type HealthResponse, type SystemStatus } from "@/lib/api";
import { formatDateTime, formatRelative } from "@/lib/utils";

export function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [h, s] = await Promise.all([Status.health(), Status.system()]);
        if (cancelled) return;
        setHealth(h);
        setSystem(s);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError((err as Error).message ?? "Failed to load");
      }
    }
    load();
    const id = setInterval(load, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const symbols = health?.supported_symbols ?? [];
  const totalRows = system
    ? Object.values(system.rows_per_symbol).reduce((acc, v) => acc + (v ?? 0), 0)
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground">Live overview of the analytics pipeline.</p>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pipeline</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {system?.pipeline_running ? "Running" : "Idle"}
            </div>
            <p className="text-xs text-muted-foreground">
              Recompute every {health?.compute_interval_seconds ?? "—"}s
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active API Keys</CardTitle>
            <KeyRound className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{system?.active_api_keys ?? "—"}</div>
            <p className="text-xs text-muted-foreground">In rotation right now</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Chain Rows</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalRows.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Across all symbols</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Databento Event</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {system?.last_databento_event ? formatRelative(system.last_databento_event) : "—"}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatDateTime(system?.last_databento_event)}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Per-Symbol Compute</CardTitle>
          <CardDescription>Most recent successful compute for each underlying.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            {symbols.length === 0 && (
              <p className="text-sm text-muted-foreground">No symbols configured.</p>
            )}
            {symbols.map((sym) => {
              const lastTs = system?.last_compute_per_symbol?.[sym];
              const dur = system?.last_compute_duration_ms?.[sym] ?? 0;
              const rows = system?.rows_per_symbol?.[sym] ?? 0;
              const metricRows = system?.metric_rows_per_symbol?.[sym] ?? 0;
              return (
                <div
                  key={sym}
                  className="flex flex-col gap-1 rounded-md border border-border bg-background/40 p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-semibold">{sym}</span>
                    <span className="text-xs text-muted-foreground">
                      {lastTs ? formatRelative(lastTs) : "no runs yet"}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 pt-2 text-xs text-muted-foreground">
                    <div>Chain: {rows.toLocaleString()}</div>
                    <div>Metrics: {metricRows.toLocaleString()}</div>
                    <div>Latency: {dur.toFixed(0)}ms</div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
