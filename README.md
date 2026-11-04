# Syscall HIDS/HIPS codebase

Kode eksperimen untuk deteksi intrusi berbasis urutan system call, evaluasi serangan mimikri, dan prototipe pencegahan eBPF. Folder ini berisi kode saja; dataset, hasil eksperimen, checkpoint, dan naskah tidak disertakan.

## Struktur

```text
Skripsi-Syscall-Codebase/
├── syscall_hids/
│   ├── data/          # persiapan ADFA-LD dan LID-DS, vocabulary, helper data
│   ├── attacks/       # mimikri statis dan serangan adaptif
│   ├── training/      # model, baseline, ablasi, dan pelatihan
│   ├── evaluation/    # evaluasi adaptif, lintas dataset, dan budget serangan
│   ├── diagnostics/   # shortcut, atribusi, artefak injeksi, analisis statistik
│   ├── prevention/    # kebijakan pencegahan, containment, latensi, Pareto
│   └── export/        # ekspor model ke ONNX
├── deploy/
│   ├── capture/       # pemantauan eBPF dan event yang hilang
│   ├── runtime/       # daemon HIPS dan vocabulary untuk deployment
│   ├── validation/    # workload serangan, no-op shim, uji fungsional
│   └── benchmarks/    # overhead dan latensi pipeline di VM
├── run_experiments.sh # urutan eksperimen inti
├── requirements.txt   # dependensi Python
├── .gitignore         # data dan artefak lokal yang tidak dipush
└── README.md
```

Skrip dalam `syscall_hids/` dijalankan sebagai modul Python dari akar proyek. Folder output tetap dibuat relatif terhadap akar proyek.

## Persiapan

Gunakan Python 3.12 dan jalankan perintah berikut dari akar proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Di Windows PowerShell, aktifkan lingkungan dengan `.venv\Scripts\Activate.ps1`. Untuk CUDA, sesuaikan instalasi PyTorch dengan perangkat dan driver host. Prototipe Linux juga membutuhkan BCC (`python3-bpfcc`) serta hak akses untuk attach eBPF dan mengirim signal ke proses.

Letakkan ADFA-LD di `archive-kaggle/ADFA-LD/` dan rekaman LID-DS 2021 di `LID-DS/`. Dataset tidak didistribusikan di folder ini. Data turunan (`prepared_data/`, `prepared_data_lidds/`, `adversarial_data/`), checkpoint (`models/`), dan output (`results/`) diabaikan oleh Git. Periksa lisensi dataset sebelum mengunduh dan menggunakannya.

## Menjalankan eksperimen

Di Git Bash atau shell Linux:

```bash
bash run_experiments.sh
```

Untuk menjalankan satu tahap saja:

```bash
python3 -m syscall_hids.data.adfa_ld_preparation
python3 -m syscall_hids.attacks.adversarial_mimicry
python3 -m syscall_hids.training.corrected_pipeline --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 --out corrected_full.json
python3 -m syscall_hids.evaluation.adaptive_eval --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 --n 200
```

Perintah `run_experiments.sh` memerlukan kedua dataset dan dapat memakan waktu serta sumber daya komputasi yang besar. Hasil dibuat di `results/` dan checkpoint di `models/`. Jalankan skrip deployment dari akar proyek agar path pada contoh tetap sesuai.

## Prototipe HIPS

`deploy/runtime/hips_daemon.py` menjalankan capture syscall via eBPF, inferensi ONNX, pemetaan skor ke tindakan, dan enforcement melalui signal userspace. `deploy/runtime/hips_daemon_v2.py` adalah varian dengan whitelist. `deploy/validation/attack_simulator.py` dan `deploy/validation/benign_workload.sh` menyediakan workload uji; skrip validasi dan `noop_shim.c` di folder yang sama memeriksa validitas fungsional injeksi no-op di VM terisolasi.

Contoh dry run di Linux, setelah mengekspor ONNX dan vocabulary dari eksperimen:

```bash
sudo env PYTHONPATH=<venv>/site-packages /usr/bin/python3 deploy/runtime/hips_daemon.py \
  --model models/a4_full.onnx --vocab prepared_data/vocabulary.json
```

## Asal dan batasan

Repositori ini memuat kode eksperimen dan prototipe deployment. Dataset, checkpoint, hasil eksperimen, serta materi penulisan dan presentasi dikecualikan oleh `.gitignore` atau tidak disertakan.
