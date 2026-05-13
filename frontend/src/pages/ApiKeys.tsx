import { Copy, Plus } from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ApiKeys, Status, type ApiKeySummary, type HealthResponse } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

interface CreateForm {
  label: string;
  symbols: string[];
  expires_at: string;
}

export function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKeySummary[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [createForm, setCreateForm] = useState<CreateForm>({
    label: "",
    symbols: [],
    expires_at: "",
  });
  const [creating, setCreating] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);

  const symbols = health?.supported_symbols ?? [];

  const refresh = useMemo(
    () => async () => {
      try {
        const [k, h] = await Promise.all([ApiKeys.list(), Status.health()]);
        setKeys(k);
        setHealth(h);
        setError(null);
      } catch (err) {
        const detail =
          (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        setError(detail || (err as Error).message);
      }
    },
    [],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onCreate(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      const result = await ApiKeys.create({
        label: createForm.label,
        allowed_symbols: createForm.symbols,
        expires_at: createForm.expires_at ? new Date(createForm.expires_at).toISOString() : null,
      });
      setNewKey(result.plaintext_key);
      setCreateOpen(false);
      setCreateForm({ label: "", symbols: [], expires_at: "" });
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function toggleActive(row: ApiKeySummary) {
    try {
      await ApiKeys.update(row.id, { is_active: !row.is_active });
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    }
  }

  async function remove(row: ApiKeySummary) {
    if (!confirm(`Revoke key "${row.label}"? This cannot be undone.`)) return;
    try {
      await ApiKeys.remove(row.id);
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    }
  }

  function copy(text: string) {
    navigator.clipboard.writeText(text).catch(() => {
      /* no-op */
    });
  }

  function statusBadge(row: ApiKeySummary) {
    if (!row.is_active) return <Badge variant="warning">inactive</Badge>;
    if (row.expires_at && new Date(row.expires_at).getTime() < Date.now()) {
      return <Badge variant="destructive">expired</Badge>;
    }
    return <Badge variant="success">active</Badge>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">API Keys</h1>
          <p className="text-sm text-muted-foreground">
            Manage end-user API keys. Keys are hashed at rest and can never be retrieved after creation.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4" /> New key
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>All keys</CardTitle>
          <CardDescription>{keys.length} total</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Label</TableHead>
                <TableHead>Prefix</TableHead>
                <TableHead>Symbols</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Expires</TableHead>
                <TableHead>Usage</TableHead>
                <TableHead>Last used</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="py-8 text-center text-muted-foreground">
                    No keys yet. Create one to get started.
                  </TableCell>
                </TableRow>
              )}
              {keys.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-medium">{row.label}</TableCell>
                  <TableCell className="font-mono text-xs">{row.key_prefix}…</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {row.allowed_symbols.map((s) => (
                        <Badge key={s} variant="outline">{s}</Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell>{statusBadge(row)}</TableCell>
                  <TableCell>{row.expires_at ? formatDateTime(row.expires_at) : "never"}</TableCell>
                  <TableCell>{row.usage_count.toLocaleString()}</TableCell>
                  <TableCell>{formatDateTime(row.last_used_at)}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button size="sm" variant="outline" onClick={() => toggleActive(row)}>
                        {row.is_active ? "Disable" : "Enable"}
                      </Button>
                      <Button size="sm" variant="destructive" onClick={() => remove(row)}>
                        Revoke
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create API key</DialogTitle>
            <DialogDescription>
              The plaintext key is shown only once after creation. Copy it now.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={onCreate} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="label">Label</Label>
              <Input
                id="label"
                value={createForm.label}
                onChange={(e) => setCreateForm({ ...createForm, label: e.target.value })}
                placeholder="e.g. ATAS prod"
                required
              />
            </div>
            <div className="space-y-2">
              <Label>Allowed symbols</Label>
              <div className="flex flex-wrap gap-3">
                {symbols.length === 0 && (
                  <span className="text-xs text-muted-foreground">No symbols configured</span>
                )}
                {symbols.map((s) => {
                  const checked = createForm.symbols.includes(s);
                  return (
                    <label key={s} className="inline-flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-border bg-background"
                        checked={checked}
                        onChange={(e) => {
                          setCreateForm((prev) => ({
                            ...prev,
                            symbols: e.target.checked
                              ? [...prev.symbols, s]
                              : prev.symbols.filter((x) => x !== s),
                          }));
                        }}
                      />
                      {s}
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="expires">Expires at (optional)</Label>
              <Input
                id="expires"
                type="datetime-local"
                value={createForm.expires_at}
                onChange={(e) => setCreateForm({ ...createForm, expires_at: e.target.value })}
              />
              <p className="text-xs text-muted-foreground">Leave blank for no expiry.</p>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setCreateOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={creating || createForm.symbols.length === 0}>
                {creating ? "Creating…" : "Create key"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={newKey !== null} onOpenChange={(open) => !open && setNewKey(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save your API key</DialogTitle>
            <DialogDescription>
              This is the only time the key will be shown. Store it in a secure location now.
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border border-border bg-muted/40 p-3 font-mono text-sm break-all">
            {newKey}
          </div>
          <DialogFooter>
            <Button onClick={() => newKey && copy(newKey)}>
              <Copy className="h-4 w-4" /> Copy
            </Button>
            <Button variant="outline" onClick={() => setNewKey(null)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
