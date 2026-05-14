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
  // Rev 4 fields aren't typed in lib/streamClient yet — accessed via a
  // narrow shape projection here so the existing typed props on
  // {GexChart, HiroPanel, WallsCards, FlowFeed, RegimeBadge} keep working.
  const revFour = (snapshot?.data ?? {}) as {
    session_state?: {
      is_rth: boolean;
      session_open: string | null;
      session_close: string | null;
      minutes_to_close: number | null;
      tau_0dte_years: number | null;
      is_expiration_day: boolean;
      symbol?: string;
    };
    spot?: {
      price: number;
      source: string;
      futures_price?: number | null;
      basis?: number | null;
      basis_age_seconds?: number | null;
      parity_deviation_pct?: number | null;
    };
    zero_dte?: {
      gex_oi?: { net_total: number };
      gex_volume?: { net_total: number };
      charm_total?: { net_total: number };
      charm_decay_rate?: number;
      flip_speed?: number;
    };
  };

  const sess = revFour.session_state;
  const spot = revFour.spot;
  const zdte = revFour.zero_dte;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight">Live</h1>
            <RegimeBadge regime={data?.regime} />
            {sess?.is_expiration_day && (
              <span className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs font-semibold text-amber-500">
                0DTE day
              </span>
            )}
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

      {/* Rev 4 — RTH session banner + spot-source badge + flip speed */}
      <div className="grid gap-3 md:grid-cols-3">
        <div
          className={`rounded-md border p-3 ${
            sess?.is_rth
              ? "border-emerald-500/40 bg-emerald-500/10"
              : "border-border bg-muted/40"
          }`}
        >
          <div className="text-xs uppercase text-muted-foreground">Session</div>
          <div className="text-sm font-semibold">
            {sess?.is_rth ? "RTH open" : sess ? "After hours" : "—"}
          </div>
          {sess?.minutes_to_close !== null && sess?.minutes_to_close !== undefined && sess.is_rth && (
            <div className="text-xs text-muted-foreground">
              {Math.max(0, Math.round(sess.minutes_to_close))} min to close
            </div>
          )}
        </div>

        <div className="rounded-md border border-border p-3">
          <div className="text-xs uppercase text-muted-foreground">Spot</div>
          <div className="text-sm font-semibold">
            {spot ? spot.price.toFixed(2) : "—"}
          </div>
          {spot && (
            <div className="text-xs text-muted-foreground">
              source: <span className="font-mono">{spot.source}</span>
              {spot.basis !== null && spot.basis !== undefined && (
                <> · basis {spot.basis.toFixed(2)}</>
              )}
            </div>
          )}
        </div>

        <div className="rounded-md border border-border p-3">
          <div className="text-xs uppercase text-muted-foreground">0DTE flip speed</div>
          <div className="text-sm font-semibold">
            {zdte?.flip_speed !== undefined ? zdte.flip_speed.toFixed(2) : "—"}
          </div>
          {zdte?.charm_decay_rate !== undefined && (
            <div className="text-xs text-muted-foreground">
              charm decay {zdte.charm_decay_rate.toFixed(4)} /hr
            </div>
          )}
        </div>
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
