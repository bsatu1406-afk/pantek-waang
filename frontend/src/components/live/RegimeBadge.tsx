import { cn } from "@/lib/utils";
import type { RegimePayload } from "@/lib/streamClient";

export type RegimeKind = "bullish" | "neutral" | "bearish";

function classifyLabel(label: string | undefined): RegimeKind {
  const v = (label ?? "").toLowerCase();
  if (v === "bullish" || v === "bull") return "bullish";
  if (v === "bearish" || v === "bear") return "bearish";
  return "neutral";
}

function pickEntry(regime: RegimePayload | undefined): { label: string; score: number } | null {
  if (!regime) return null;
  if (regime.label) {
    return { label: String(regime.label), score: Number(regime.score ?? 0) };
  }
  if (regime.oi) return { label: regime.oi.label, score: regime.oi.score };
  if (regime.vol) return { label: regime.vol.label, score: regime.vol.score };
  return null;
}

export interface RegimeBadgeProps {
  regime: RegimePayload | undefined;
}

export function RegimeBadge({ regime }: RegimeBadgeProps) {
  const entry = pickEntry(regime);
  const kind = classifyLabel(entry?.label);
  const text = entry ? entry.label.toUpperCase() : "UNKNOWN";

  return (
    <div
      data-testid="regime-badge"
      data-kind={kind}
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide",
        kind === "bullish" && "bg-emerald-500/15 text-emerald-400",
        kind === "neutral" && "bg-amber-500/15 text-amber-300",
        kind === "bearish" && "bg-red-500/15 text-red-400",
      )}
    >
      <span
        className={cn(
          "inline-block h-2 w-2 rounded-full",
          kind === "bullish" && "bg-emerald-400",
          kind === "neutral" && "bg-amber-300",
          kind === "bearish" && "bg-red-400",
        )}
      />
      {text}
      {entry && (
        <span className="ml-1 text-[0.7rem] font-normal text-muted-foreground">
          {entry.score.toFixed(2)}
        </span>
      )}
    </div>
  );
}
