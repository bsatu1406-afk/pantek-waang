import { useMemo } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { MaxPainPayload, WallsPayload, WallStrike } from "@/lib/streamClient";

function formatStrike(value: number): string {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatBig(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}

function topN(rows: WallStrike[] | undefined, n: number): WallStrike[] {
  if (!rows) return [];
  return [...rows].sort((a, b) => (a.rank || 999) - (b.rank || 999)).slice(0, n);
}

interface WallsCardProps {
  title: string;
  description: string;
  rows: WallStrike[];
  accent: "call" | "put";
}

function WallsCard({ title, description, rows, accent }: WallsCardProps) {
  const colorClass = accent === "call" ? "text-emerald-400" : "text-red-400";
  return (
    <Card>
      <CardHeader className="space-y-1 pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <div className="text-sm text-muted-foreground">No data yet.</div>
        ) : (
          <ul className="space-y-2 text-sm">
            {rows.map((r) => (
              <li key={`${r.rank}-${r.strike}`} className="flex items-center justify-between">
                <span className={`font-mono ${colorClass}`}>{formatStrike(r.strike)}</span>
                <span className="font-mono text-muted-foreground">{formatBig(r.value)}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

export interface WallsCardsProps {
  walls: WallsPayload | undefined;
  maxPain: MaxPainPayload | undefined;
}

export function WallsCards({ walls, maxPain }: WallsCardsProps) {
  const callRows = useMemo(() => topN(walls?.call_wall_oi, 5), [walls]);
  const putRows = useMemo(() => topN(walls?.put_wall_oi, 5), [walls]);

  const aggregate = maxPain?.aggregate ?? null;
  const perExpiry = (maxPain?.per_expiry ?? []).slice(0, 5);

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <WallsCard
        title="Call walls"
        description="Top OI strikes (resistance)"
        rows={callRows}
        accent="call"
      />
      <WallsCard
        title="Put walls"
        description="Top OI strikes (support)"
        rows={putRows}
        accent="put"
      />
      <Card>
        <CardHeader className="space-y-1 pb-3">
          <CardTitle className="text-base">Max pain</CardTitle>
          <CardDescription>Aggregate + nearest expiries</CardDescription>
        </CardHeader>
        <CardContent>
          {aggregate ? (
            <div className="mb-3">
              <div className="text-xs text-muted-foreground">Aggregate strike</div>
              <div className="text-xl font-semibold text-amber-300">
                {formatStrike(aggregate.strike)}
              </div>
              <div className="text-xs text-muted-foreground">
                Pain {formatBig(aggregate.value)}
              </div>
            </div>
          ) : (
            <div className="mb-3 text-sm text-muted-foreground">No aggregate yet.</div>
          )}
          {perExpiry.length === 0 ? (
            <div className="text-sm text-muted-foreground">No per-expiry data yet.</div>
          ) : (
            <ul className="space-y-2 text-sm">
              {perExpiry.map((r) => (
                <li key={r.expiration} className="flex items-center justify-between">
                  <span className="font-mono text-muted-foreground">{r.expiration}</span>
                  <span className="font-mono">{formatStrike(r.strike)}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
