import { useEffect, useState } from "react";
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
import { Status, type HealthResponse, type SystemStatus } from "@/lib/api";
import { formatDateTime, formatRelative } from "@/lib/utils";

export function SystemStatusPage() {
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
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const symbols = health?.supported_symbols ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">System Status</h1>
        <p className="text-sm text-muted-foreground">Live pipeline + ingestion telemetry.</p>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Pipeline</CardTitle>
          <CardDescription>
            Recompute interval: every {health?.compute_interval_seconds ?? "—"} seconds
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-md border border-border bg-background/40 p-4">
              <div className="text-xs text-muted-foreground">Pipeline status</div>
              <div className="text-lg font-semibold">
                {system?.pipeline_running ? "Running" : "Idle"}
              </div>
            </div>
            <div className="rounded-md border border-border bg-background/40 p-4">
              <div className="text-xs text-muted-foreground">Last Databento event</div>
              <div className="text-lg font-semibold">
                {system?.last_databento_event ? formatRelative(system.last_databento_event) : "—"}
              </div>
              <div className="text-xs text-muted-foreground">
                {formatDateTime(system?.last_databento_event)}
              </div>
            </div>
            <div className="rounded-md border border-border bg-background/40 p-4">
              <div className="text-xs text-muted-foreground">Active API keys</div>
              <div className="text-lg font-semibold">{system?.active_api_keys ?? "—"}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Per-symbol</CardTitle>
          <CardDescription>Compute latency, last run, and row counts.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Last compute</TableHead>
                <TableHead>Latency (ms)</TableHead>
                <TableHead>Chain rows</TableHead>
                <TableHead>Metric rows</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {symbols.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    No symbols configured.
                  </TableCell>
                </TableRow>
              )}
              {symbols.map((sym) => (
                <TableRow key={sym}>
                  <TableCell className="font-mono">{sym}</TableCell>
                  <TableCell>
                    {system?.last_compute_per_symbol?.[sym]
                      ? `${formatRelative(system.last_compute_per_symbol[sym])} (${formatDateTime(
                          system.last_compute_per_symbol[sym],
                        )})`
                      : "—"}
                  </TableCell>
                  <TableCell>{system?.last_compute_duration_ms?.[sym]?.toFixed(0) ?? "—"}</TableCell>
                  <TableCell>{(system?.rows_per_symbol?.[sym] ?? 0).toLocaleString()}</TableCell>
                  <TableCell>{(system?.metric_rows_per_symbol?.[sym] ?? 0).toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
