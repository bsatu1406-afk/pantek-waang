import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { ConnectionStatusIndicator } from "@/components/live/ConnectionStatus";
import { FlowFeed } from "@/components/live/FlowFeed";
import { GexChart } from "@/components/live/GexChart";
import { HiroPanel } from "@/components/live/HiroPanel";
import { RegimeBadge } from "@/components/live/RegimeBadge";
import { WallsCards } from "@/components/live/WallsCards";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Status } from "@/lib/api";
import { LiveSnapshotProvider, useLiveSnapshot } from "@/lib/streamClient";

function LiveDashboardInner() {
  const { symbol, apiKey, setSymbol, setApiKey, snapshot, status, lastFrameAt } =
    useLiveSnapshot();
  const [supportedSymbols, setSupportedSymbols] = useState<string[]>([]);
  const [draftSymbol, setDraftSymbol] = useState<string>(symbol);
  const [draftKey, setDraftKey] = useState<string>(apiKey);

  useEffect(() => {
    let cancelled = false;
    Status.health()
      .then((h) => {
        if (!cancelled) setSupportedSymbols(h.supported_symbols ?? []);
      })
      .catch(() => {
        /* health is admin-only; ignore failures */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const data = snapshot?.data;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">Live</h1>
            <RegimeBadge regime={data?.regime} />
          </div>
          <p className="text-sm text-muted-foreground">
            Streaming snapshot for <span className="font-mono">{symbol || "—"}</span>
            {snapshot?.computed_at && (
              <>
                {" "}
                · last frame {new Date(snapshot.computed_at).toLocaleTimeString()}
              </>
            )}
          </p>
        </div>
        <ConnectionStatusIndicator status={status} lastFrameAt={lastFrameAt} />
      </div>

      <div className="rounded-md border border-border bg-background/40 p-3">
        <div className="grid gap-3 md:grid-cols-3 md:items-end">
          <div>
            <Label htmlFor="symbol">Symbol</Label>
            <Input
              id="symbol"
              list="symbol-list"
              className="mt-1 font-mono"
              value={draftSymbol}
              placeholder="SPXW"
              onChange={(e) => setDraftSymbol(e.target.value.toUpperCase())}
            />
            <datalist id="symbol-list">
              {supportedSymbols.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
          </div>
          <div>
            <Label htmlFor="api-key">API key</Label>
            <Input
              id="api-key"
              type="password"
              className="mt-1 font-mono"
              value={draftKey}
              placeholder="ofa_…"
              onChange={(e) => setDraftKey(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => {
                setSymbol(draftSymbol);
                setApiKey(draftKey);
              }}
              className="gap-2"
            >
              <Activity className="h-4 w-4" />
              Connect
            </Button>
            {(symbol || apiKey) && (
              <Button
                variant="outline"
                onClick={() => {
                  setSymbol("");
                  setApiKey("");
                  setDraftSymbol("");
                  setDraftKey("");
                }}
              >
                Disconnect
              </Button>
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <GexChart payload={data?.gex} title="GEX (OI)" description="Per-strike gamma exposure" />
        </div>
        <HiroPanel payload={data?.hiro} />
      </div>

      <WallsCards walls={data?.walls} maxPain={data?.max_pain} />

      <FlowFeed flow={data?.flow} />
    </div>
  );
}

export function LivePage() {
  return (
    <LiveSnapshotProvider initialSymbol="SPXW">
      <LiveDashboardInner />
    </LiveSnapshotProvider>
  );
}
