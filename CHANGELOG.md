# Diagnostic + last_price spot-fallback patch

## Tujuan

Memperbaiki bug "all metrics = 0.00 / no Zero Gamma / no underlying" di Data
Inspector. Akar penyebabnya: chain rows tidak terisi `bid` / `ask` (kemungkinan
besar plan OPRA Pillar tidak menyertakan schema `cmbp-1` untuk NBBO updates,
sehingga ingester hanya menerima `definition` + `trades` + `statistics`). Tanpa
`bid` / `ask`, put-call parity gagal, `underlying_price` tetap null, BSM tidak
bisa hitung greek, semua metric jatuh ke 0.

## Perubahan

### Backend

1. **`backend/app/processing/spot.py`** — Sintesis spot via put-call parity
   sekarang fallback ke `last_price` ketika `bid` / `ask` tidak ada. Selama
   ada trade prints di chain, spot tetap bisa direkonstruksi.

2. **`backend/app/processing/pipeline.py`** — Logging diagnostik:
   - `pipeline_no_underlying` (warning) saat semua row tidak punya
     `underlying_price` (artinya sintesis gagal total).
   - `pipeline_low_greek_coverage` saat IV / gamma 0%.
   - Dengan begini bug 0-metrics tidak silent lagi — muncul di
     `docker logs ofa-backend`.

3. **`backend/app/ingestion/databento_live.py`** — Diagnostics surface:
   - `_cumulative_record_counts` (tidak di-clear) untuk Data Inspector.
   - `_dropped_schemas`, `_connection_attempts`, `_last_error`,
     `_first_record_at`, `_last_record_at`.
   - `_sample_record_attrs` — capture attribute snapshot record pertama
     per type, sangat berguna untuk debug parsing field bid/ask.
   - Method `diagnostics()` yang mengembalikan snapshot lengkap.

4. **`backend/app/ingestion/databento_globex.py`** — Diagnostics surface
   yang sama untuk Globex MDP 3.0 ingester.

5. **`backend/app/api/endpoints/inspector.py`** — Endpoint
   `/admin/inspector` sekarang juga mengembalikan:
   - `chain_quality[]`: per-symbol % kolom yang terisi (`bid`, `ask`,
     `last_price`, `iv`, `delta`, `gamma`, `oi`, `volume`,
     `underlying_price`) di window 1 jam terakhir, atau seluruh tabel
     kalau market tutup.
   - `ingesters.opra` dan `ingesters.globex`: registry size, schemas
     active/dropped, record counts kumulatif, last error, dst.

6. **`backend/tests/test_processing_spot.py`** — 2 test baru untuk
   memvalidasi fallback `last_price`.

### Frontend

7. **`frontend/src/lib/api.ts`** — Type baru:
   `InspectorChainQuality`, `InspectorIngesterDiag`, plus extension
   `InspectorPayload`.

8. **`frontend/src/pages/DataInspector.tsx`** — Tambahan section:
   - **Pipeline diagnostic banner** (merah) — muncul otomatis saat ada
     simbol dengan 0% gamma atau 0% underlying coverage.
   - **Chain data quality** — tabel coverage % per kolom per simbol.
     Hijau ≥95%, kuning ≥50%, merah <50%. Ini panel yang langsung
     menunjukkan apakah upstream feed mengirim quote atau tidak.
   - **Live ingesters** — per ingester (OPRA / Globex):
     - Registry size, book size, connection attempts.
     - Schemas active vs schemas dropped.
     - Record counts cumulative by message type — kalau kamu lihat
       hanya `TradeMsg`, `StatMsg`, `InstrumentDefMsg` tanpa `CMBP1Msg`,
       artinya plan OPRA-mu tidak include cmbp-1.
     - Last error message.
     - First/last record timestamp.

## Cara apply

Extract zip di folder `optionsflow-platform/` (overwrite file lama):

```
cd C:\Users\ollama\Downloads\optionsflow-platform
# extract diagnostics-update.zip ke folder ini
```

Rebuild backend + frontend (frontend rebuild perlu karena `tsc` + `vite build`
dijalankan di Dockerfile):

```
docker compose stop backend frontend
docker compose build --no-cache backend frontend
docker compose up -d backend frontend
```

Buka `http://localhost:3000/data-inspector`.

## Cara baca

1. **Banner merah "Pipeline diagnostic"** — kalau muncul, baca isinya. Itu
   memberi tahu kamu kemungkinan akar masalah.

2. **Section "Chain data quality"** — perhatikan kolom `bid%` dan `ask%`:
   - Kalau **bid% = 0% dan ask% = 0%** tapi `last%` > 0% → ingester tidak
     menerima cmbp-1 NBBO. Plan OPRA Pillar yang kamu pakai sepertinya
     tidak include schema ini. **Setelah patch ini, sintesis spot akan
     fallback ke `last_price` dan greek tetap bisa dihitung selama ada
     trade.**
   - Kalau **bid%, ask%, last%, semuanya 0%** → ingester tidak menerima
     apa-apa dari OPRA. Cek section "Live ingesters" di bawah.
   - Kalau **bid% / ask% > 80% dan greek/spot masih 0%** → ada bug lain,
     lapor balik (sertakan screenshot inspector page-nya).

3. **Section "Live ingesters"** — dua kotak (OPRA & GLOBEX):
   - **Schemas dropped** harus `(none)` (hijau). Kalau ada `cmbp-1` di
     daftar ini, gateway Databento menolak schema itu untuk plan kamu.
     Lapor balik dan kita lihat apakah ada upgrade plan atau alternatif.
   - **Cumulative record counts**: harus ada `CMBP1Msg` (atau setidaknya
     `TradeMsg`) untuk OPRA, dan `MBP10Msg` / `TradeMsg` untuk Globex.
     Kalau cuma `InstrumentDefMsg` yang ada, ingester baru handshake tapi
     belum ada data flow.
   - **Last error**: kalau ada pesan error di sini, paste balik supaya bisa
     di-trace.

## Dampak
- Setelah patch, **kalau market US open dan ada trade tape**, spot akan
  ter-recovery dari last_price → BSM bisa hitung greek → semua metric
  populate. Zero Gamma, top long-gamma, top short-gamma akan keluar.
- **Kalau market tutup**, last_price stale dan trade tape kosong — wajar
  metric tetap 0. Tapi inspector page sekarang akan menampilkan diagnostic
  jelas (pipeline_no_underlying warning di log + banner di UI).
- Tidak mengubah logika perhitungan apa-apa (BSM, GEX, Vanna, Charm, dst).
  Hanya menambah fallback path & diagnostic surface.
