import { Plus } from "lucide-react";
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
import {
  DatabentoKeys,
  type DatabentoDataset,
  type DatabentoKeySummary,
} from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

const DATASETS: DatabentoDataset[] = ["OPRA.PILLAR", "GLBX.MDP3", "BOTH"];

interface CreateForm {
  label: string;
  dataset: DatabentoDataset;
  api_key: string;
  priority: number;
}

function statusBadge(row: DatabentoKeySummary) {
  if (!row.is_active) return <Badge variant="warning">disabled</Badge>;
  if (row.error_count > 0 && row.last_error_at) {
    const ageMs = Date.now() - new Date(row.last_error_at).getTime();
    if (ageMs < 60 * 60 * 1_000) {
      return <Badge variant="destructive">error</Badge>;
    }
    return <Badge variant="warning">recovering</Badge>;
  }
  return <Badge variant="success">active</Badge>;
}

export function DatabentoKeysPage() {
  const [rows, setRows] = useState<DatabentoKeySummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<CreateForm>({
    label: "",
    dataset: "OPRA.PILLAR",
    api_key: "",
    priority: 100,
  });
  const [submitting, setSubmitting] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const refresh = useMemo(
    () => async () => {
      try {
        const r = await DatabentoKeys.list();
        setRows(r);
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
    setSubmitting(true);
    setError(null);
    try {
      await DatabentoKeys.create({
        label: form.label.trim(),
        dataset: form.dataset,
        api_key: form.api_key.trim(),
        priority: form.priority,
      });
      setOpen(false);
      setForm({ label: "", dataset: "OPRA.PILLAR", api_key: "", priority: 100 });
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  async function toggleActive(row: DatabentoKeySummary) {
    try {
      await DatabentoKeys.update(row.id, { is_active: !row.is_active });
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    }
  }

  async function changePriority(row: DatabentoKeySummary, delta: number) {
    try {
      await DatabentoKeys.update(row.id, {
        priority: Math.max(0, row.priority + delta),
      });
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    }
  }

  async function remove(row: DatabentoKeySummary) {
    if (
      !confirm(
        `Delete Databento key "${row.label}"? This will remove it from the failover pool.`,
      )
    ) {
      return;
    }
    try {
      await DatabentoKeys.remove(row.id);
      await refresh();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(detail || (err as Error).message);
    }
  }

  async function testKey(row: DatabentoKeySummary) {
    try {
      const r = await DatabentoKeys.test(row.id);
      setTestResult(`${row.label}: ${r.ok ? "OK" : "FAILED"} — ${r.message}`);
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setTestResult(detail || (err as Error).message);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Databento Keys</h1>
          <p className="text-sm text-muted-foreground">
            Failover pool for the OPRA.PILLAR &amp; GLBX.MDP3 live feeds. Keys are
            encrypted at rest with a key derived from <code>JWT_SECRET</code>.
            The ingester tries env-configured keys first, then DB keys ordered
            by priority ASC.
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="h-4 w-4" /> Add key
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}
      {testResult && (
        <div className="rounded-md border border-border bg-muted p-3 text-sm">
          {testResult}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Pool</CardTitle>
          <CardDescription>{rows.length} key(s) registered</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Label</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Prefix</TableHead>
                <TableHead>Priority</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Last used</TableHead>
                <TableHead>Last error</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-muted-foreground">
                    No keys yet. Add one to enable failover.
                  </TableCell>
                </TableRow>
              )}
              {rows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-medium">{row.label}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{row.dataset}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{row.api_key_prefix}…</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => changePriority(row, -10)}
                        title="Higher priority (lower number)"
                      >
                        ↑
                      </Button>
                      <span className="w-10 text-center font-mono">{row.priority}</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => changePriority(row, 10)}
                        title="Lower priority (higher number)"
                      >
                        ↓
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell>{statusBadge(row)}</TableCell>
                  <TableCell>{formatDateTime(row.last_used_at)}</TableCell>
                  <TableCell className="max-w-xs text-xs">
                    {row.last_error_msg ? (
                      <span title={row.last_error_msg}>
                        {row.last_error_msg.slice(0, 60)}
                        {row.last_error_msg.length > 60 ? "…" : ""}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="space-x-2">
                    <Button variant="outline" size="sm" onClick={() => testKey(row)}>
                      Test
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => toggleActive(row)}
                    >
                      {row.is_active ? "Disable" : "Enable"}
                    </Button>
                    <Button variant="destructive" size="sm" onClick={() => remove(row)}>
                      Delete
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <form onSubmit={onCreate} className="space-y-4">
            <DialogHeader>
              <DialogTitle>Add Databento Key</DialogTitle>
              <DialogDescription>
                The plaintext key is encrypted before storage and never shown again.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-2">
              <Label htmlFor="label">Label</Label>
              <Input
                id="label"
                placeholder="e.g. Backup OPRA #1"
                value={form.label}
                onChange={(e) => setForm({ ...form, label: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="dataset">Dataset</Label>
              <select
                id="dataset"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={form.dataset}
                onChange={(e) =>
                  setForm({ ...form, dataset: e.target.value as DatabentoDataset })
                }
              >
                {DATASETS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                <code>BOTH</code> covers both OPRA &amp; GLBX with a single subscription.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="api_key">API Key</Label>
              <Input
                id="api_key"
                type="password"
                placeholder="db-..."
                value={form.api_key}
                onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="priority">Priority (lower = tried first)</Label>
              <Input
                id="priority"
                type="number"
                min={0}
                max={10000}
                value={form.priority}
                onChange={(e) =>
                  setForm({ ...form, priority: Number(e.target.value) })
                }
              />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? "Adding…" : "Add key"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
