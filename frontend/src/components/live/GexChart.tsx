import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
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
import type { GexPayload } from "@/lib/streamClient";

function formatStrike(value: number): string {
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  return value.toString();
}

function formatGex(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}

export interface GexChartProps {
  payload: GexPayload | undefined;
  title?: string;
  description?: string;
}

export function GexChart({ payload, title = "GEX", description }: GexChartProps) {
  const data = useMemo(() => {
    const curve = payload?.curve ?? [];
    return [...curve]
      .filter((p) => Number.isFinite(p.strike) && Number.isFinite(p.net_gex))
      .sort((a, b) => a.strike - b.strike)
      .map((p) => ({ strike: p.strike, gex: p.net_gex }));
  }, [payload]);

  const zeroGamma = payload?.zero_gamma ?? null;
  const underlying = payload?.underlying_price ?? null;
  const netTotal = payload?.net_total ?? 0;

  return (
    <Card data-testid="gex-chart-card">
      <CardHeader className="space-y-1">
        <div className="flex items-center justify-between gap-2">
          <CardTitle>{title}</CardTitle>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">Net total</div>
            <div
              className={
                netTotal >= 0
                  ? "text-base font-semibold text-emerald-400"
                  : "text-base font-semibold text-red-400"
              }
            >
              {formatGex(netTotal)}
            </div>
          </div>
        </div>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No GEX data yet.
          </div>
        ) : (
          <div className="h-64 w-full" data-testid="gex-chart-body">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="gexPositive" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34d399" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#34d399" stopOpacity={0.05} />
                  </linearGradient>
                  <linearGradient id="gexNegative" x1="0" y1="1" x2="0" y2="0">
                    <stop offset="0%" stopColor="#f87171" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#f87171" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="hsl(215 28% 17%)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="strike"
                  tickFormatter={formatStrike}
                  stroke="hsl(215 20% 65%)"
                  fontSize={11}
                />
                <YAxis
                  tickFormatter={formatGex}
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
                  labelFormatter={(v) => `Strike ${formatStrike(Number(v))}`}
                  formatter={(v: number) => [formatGex(Number(v)), "GEX"]}
                />
                <ReferenceLine y={0} stroke="hsl(215 20% 65%)" strokeDasharray="2 2" />
                {zeroGamma != null && Number.isFinite(zeroGamma) && (
                  <ReferenceLine
                    x={zeroGamma}
                    stroke="#38bdf8"
                    strokeDasharray="3 3"
                    label={{ value: "0γ", fill: "#38bdf8", position: "top", fontSize: 11 }}
                  />
                )}
                {underlying != null && Number.isFinite(underlying) && (
                  <ReferenceLine
                    x={underlying}
                    stroke="#facc15"
                    strokeDasharray="3 3"
                    label={{ value: "Spot", fill: "#facc15", position: "top", fontSize: 11 }}
                  />
                )}
                <Area
                  type="monotone"
                  dataKey="gex"
                  stroke="#34d399"
                  fill="url(#gexPositive)"
                  isAnimationActive={false}
                  baseValue={0}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
