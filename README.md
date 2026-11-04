# Syscall HIDS/HIPS

Kode, checkpoint, data olahan, dan hasil evaluasi untuk eksperimen syscall-sequence HIDS/HIPS serta prototipe pencegahan eBPF.

## Struktur

| Folder | Kegunaan |
|---|---|
| `syscall_hids/data/` | Persiapan ADFA-LD dan LID-DS, vocabulary, helper data |
| `syscall_hids/attacks/` | Mimikri statis dan serangan adaptif |
| `syscall_hids/training/` | Model, baseline, ablasi, dan pelatihan |
| `syscall_hids/evaluation/` | Evaluasi adaptif dan lintas dataset |
| `syscall_hids/diagnostics/` | Analisis shortcut, atribusi, artefak injeksi, statistik |
| `syscall_hids/prevention/` | Kebijakan pencegahan, containment, dan latensi |
| `syscall_hids/export/` | Ekspor model ke ONNX |
| `deploy/` | Capture eBPF, daemon HIPS, validasi, dan benchmark VM |
| `models/` | Checkpoint PyTorch dan model ONNX |
| `prepared_data/`, `prepared_data_lidds/` | Data olahan dan vocabulary untuk evaluasi |
| `adversarial_data/` | Sampel mimikri statis hasil generator |
| `results/` | Laporan JSON, split validasi, dan log eksperimen |

Model `C_*_seed*.pt` adalah checkpoint dari pipeline terkoreksi. Tag lama seperti `A4_full_seed*.pt` dipertahankan untuk skrip diagnostik dan ekspor. `models/a4_full.onnx` dan `models/encoder_best.pt` adalah artefak A4 seed 42. File `*.pre_fix.pt` adalah checkpoint historis sebelum koreksi; jangan gunakan sebagai model utama.

## Persiapan

Jalankan perintah dari akar repositori dengan Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Di Windows PowerShell, aktifkan lingkungan dengan `.venv\Scripts\Activate.ps1`. Untuk CUDA, sesuaikan instalasi PyTorch dengan perangkat dan driver host. Prototipe Linux memerlukan BCC (`python3-bpfcc`) dari paket OS serta izin untuk attach eBPF dan mengirim signal ke proses.

Checkpoint dan data olahan disertakan agar analisis yang memakai artefak tersimpan dapat dijalankan tanpa menyiapkan dataset mentah. Untuk mengulang ekstraksi dan pelatihan dari awal, letakkan ADFA-LD di `archive-kaggle/ADFA-LD/` serta rekaman LID-DS 2021 di `LID-DS/`. Dataset mentah tersebut tidak disertakan. Periksa ketentuan distribusi dataset sebelum memublikasikan data olahannya.

## Menjalankan eksperimen

Skrip Python dalam `syscall_hids/` dijalankan sebagai modul dari akar repositori. Contoh:

```bash
python3 -m syscall_hids.evaluation.adaptive_eval --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 --n 200
python3 -m syscall_hids.diagnostics.analyze_corrected
```

Untuk menjalankan eksperimen dari data olahan yang disertakan, gunakan lingkungan Python eksperimen dengan dependensi di `requirements.txt`:

```bash
bash run_experiments.sh
```

Jika ADFA-LD mentah tersimpan di lokasi lain, jalankan `ADFA_LD_ROOT=/lokasi/ADFA-LD bash run_experiments.sh`. Pilih direktori yang langsung berisi `Training_Data_Master`, `Validation_Data_Master`, dan `Attack_Data_Master`. Tanpa trace mentah, runner melewati pembuatan data, uji rasio injeksi, dan containment per proses. Untuk LID-DS, runner memakai rekaman mentah, cache ekstraksi, atau data olahan yang tersedia. Eksperimen melatih model dan dapat menulis ulang checkpoint dan hasil. Untuk deteksi langsung di VM, gunakan daemon pada bagian berikut.

## Prototipe HIPS

`deploy/runtime/hips_daemon.py` menangkap syscall via eBPF, menjalankan inferensi ONNX, menentukan tindakan, dan dapat melakukan enforcement melalui signal userspace. Varian `hips_daemon_v2.py` menambahkan whitelist. Skrip di `deploy/validation/` menyediakan workload dan pemeriksaan no-op; `deploy/benchmarks/` mengukur overhead dan latensi.

Untuk menjalankan deteksi di Linux x86_64, siapkan satu model ONNX dan vocabulary yang memetakan nomor syscall host ke token model. Daemon membaca event langsung dari eBPF. Aktifkan environment runtime yang berisi BCC, NumPy, dan ONNX Runtime. 