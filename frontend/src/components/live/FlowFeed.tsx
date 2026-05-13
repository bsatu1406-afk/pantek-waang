import { useMemo } from "react";
import { ArrowDown, ArrowUp, Layers, Megaphone, Zap } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { FlowEvent, FlowPayload } from "@/lib/streamClient";

function formatPremium(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

function formatTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

function contractLabel(ev: FlowEvent): string {
  if (ev.contract_label) return ev.contract_label;
  const parts = [ev.symbol, ev.expiration, ev.strike, ev.option_type]
    .filter((v) => v !== undefined && v !== null && v !== "")
    .map((v) => String(v));
  return parts.length > 0 ? parts.join(" ") : "—";
}

function eventStyle(eventType: string): { icon: typeof Zap; variant: "default" | "secondary" | "warning" } {
  const t = eventType.toUpperCase();
  if (t === "SWEEP") return { icon: Zap, variant: "warning" };
  if (t === "BLOCK") return { icon: Layers, variant: "secondary" };
  return { icon: Megaphone, variant: "default" };
}

function sideStyle(side: number): { label: string; className: string; icon: typeof ArrowUp } {
  if (side > 0) {
    return { label: "BUY", className: "text-emerald-400", icon: ArrowUp };
  }
  if (side < 0) {
    return { label: "SELL", className: "text-red-400", icon: ArrowDown };
  }
  return { label: "MID", className: "text-muted-foreground", icon: ArrowUp };
}

export interface FlowFeedProps {
  flow: FlowPayload | undefined;
}

export function FlowFeed({ flow }: FlowFeedProps) {
  const events = useMemo(() => {
    const list = flow?.events ?? [];
    return [...list]
      .sort((a, b) => {
        const ta = new Date(a.ts).getTime();
        const tb = new Date(b.ts).getTime();
        if (Number.isNaN(ta) && Number.isNaN(tb)) return 0;
        if (Number.isNaN(ta)) return 1;
        if (Number.isNaN(tb)) return -1;
        return tb - ta;
      })
      .slice(0, 50);
  }, [flow]);

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle>Flow feed</CardTitle>
        <CardDescription>Recent SWEEP / BLOCK / UOA events</CardDescription>
      </CardHeader>
      <CardContent>
        {events.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
            No flow events yet.
          </div>
        ) : (
          <ul
            className="max-h-96 space-y-2 overflow-y-auto pr-1"
            data-testid="flow-feed-list"
          >
            {events.map((ev, idx) => {
              const { icon: Icon, variant } = eventStyle(ev.event_type);
              const side = sideStyle(ev.side);
              const SideIcon = side.icon;
              return (
                <li
                  key={ev.id ?? `${ev.ts}-${idx}`}
                  className="flex items-center gap-3 rounded-md border border-border/40 bg-background/40 px-3 py-2 text-sm"
                >
                  <Badge variant={variant} className="gap-1">
                    <Icon className="h-3 w-3" />
                    {ev.event_type.toUpperCase()}
                  </Badge>
                  <div className="flex flex-1 flex-col">
                    <span className="font-mono text-xs text-muted-foreground">
                      {formatTime(ev.ts)}
                    </span>
                    <span className="truncate font-mono">{contractLabel(ev)}</span>
                  </div>
                  <div className={cn("flex items-center gap-1 text-xs", side.className)}>
                    <SideIcon className="h-3 w-3" />
                    {side.label}
                  </div>
                  <div className="w-20 text-right font-mono text-sm">
                    {formatPremium(ev.premium)}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
