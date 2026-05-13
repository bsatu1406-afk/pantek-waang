import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { HiroPayload } from "@/lib/streamClient";

function formatValue(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return ts;
  }
}

export interface HiroPanelProps {
  payload: HiroPayload | undefined;
}

export function HiroPanel({ payload }: HiroPanelProps) {
  const data = useMemo(() => {
    const series = payload?.series ?? [];
    return series
      .filter((p) => p && Number.isFinite(p.value))
      .map((p) => ({ ts: p.ts, value: p.value }));
  }, [payload]);

  const cumulative = payload?.cumulative ?? 0;
  const positive = cumulative >= 0;

  return (
    <Card data-testid="hiro-panel-card">
      <CardHeader className="space-y-1">
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle>HIRO</CardTitle>
            <CardDescription>
              Signed-premium cumulative ({payload?.bucket_size ?? "—"})
            </CardDescription>
          </div>
          <div
            className={cn(
              "rounded-full px-3 py-1 text-sm font-semibold tabular-nums",
              positive
                ? "bg-emerald-500/15 text-emerald-400"
                : "bg-red-500/15 text-red-400",
            )}
            data-testid="hiro-badge"
          >
            {positive ? "+" : ""}
            {formatValue(cumulative)}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
            No HIRO data yet.
          </div>
        ) : (
          <div className="h-40 w-full" data-testid="hiro-chart-body">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="hsl(215 28% 17%)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="ts"
                  tickFormatter={formatTime}
                  stroke="hsl(215 20% 65%)"
                  fontSize={11}
                  minTickGap={32}
                />
                <YAxis
                  tickFormatter={formatValue}
                  stroke="hsl(215 20% 65%)"
                  fontSize={11}
                  width={56}
                />
                <Tooltip
                  contentStyle={{
                    background: "hsl(222 47% 7%)",
                    border: "1px solid hsl(215 28% 17%)",
                    fontSize: 12,
                  }}
                  labelFormatter={(v) => formatTime(String(v))}
                  formatter={(v: number) => [formatValue(Number(v)), "Cumulative"]}
                />
                <ReferenceLine y={0} stroke="hsl(215 20% 65%)" strokeDasharray="2 2" />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={positive ? "#34d399" : "#f87171"}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
