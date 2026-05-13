import { cn } from "@/lib/utils";
import type { ConnectionStatus as Status } from "@/lib/streamClient";

const LABELS: Record<Status, string> = {
  open: "Connected",
  connecting: "Connecting",
  reconnecting: "Reconnecting",
  closed: "Disconnected",
  error: "Error",
};

function colorFor(status: Status): string {
  switch (status) {
    case "open":
      return "bg-emerald-500";
    case "connecting":
    case "reconnecting":
      return "bg-amber-400";
    case "error":
    case "closed":
    default:
      return "bg-red-500";
  }
}

export interface ConnectionStatusIndicatorProps {
  status: Status;
  lastFrameAt?: number | null;
}

export function ConnectionStatusIndicator({
  status,
  lastFrameAt,
}: ConnectionStatusIndicatorProps) {
  const color = colorFor(status);
  const label = LABELS[status];
  return (
    <div
      data-testid="connection-status"
      data-status={status}
      className="inline-flex items-center gap-2 rounded-md border border-border bg-background/60 px-2 py-1 text-xs"
    >
      <span
        className={cn("inline-block h-2 w-2 rounded-full", color, status === "open" && "animate-pulse")}
        aria-hidden
      />
      <span className="font-medium">{label}</span>
      {lastFrameAt && (
        <span className="font-mono text-muted-foreground">
          {new Date(lastFrameAt).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })}
        </span>
      )}
    </div>
  );
}
