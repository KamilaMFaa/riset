# Dashboard Identifikasi dan Analisis Tren Isu Kesehatan Masyarakat di Bangkalan
## Menggunakan SBERT-BERTopic

Proyek penelitian mahasiswa Teknik Informatika. Dashboard ini akan mengidentifikasi
dan menganalisis tren isu kesehatan masyarakat di Kabupaten Bangkalan berdasarkan
data berita dan ulasan Google Maps, menggunakan pipeline SBERT (embedding) +
BERTopic (topic modeling) + HDBSCAN (clustering), dengan hasil akhir ditampilkan
di dashboard Streamlit.

> Fokus penelitian adalah analisis isu kesehatan masyarakat, **bukan** sistem
> diagnosis medis.

## Struktur Folder

```
project/
│
├── data/
│   ├── raw/                # Dataset mentah, TIDAK PERNAH ditimpa
│   │   ├── berita_kesehatan_bangkalan_10tahun.csv
│   │   └── ulasan_kesehatan_lingkungan_bangkalan.csv
│   │
│   ├── cleaned/            # Output notebook 01 (audit & cleaning)
│   │   ├── berita_bangkalan_cleaned.csv
│   │   ├── ulasan_bangkalan_cleaned.csv
│   │   ├── cleaning_audit.csv
│   │   └── cleaning_log.json
│   │
│   ├── validation/         # Output notebook 01b (validasi relevansi)
│   │   ├── berita_manual_review.xlsx / .csv
│   │   ├── berita_sample_check.xlsx
│   │   ├── review_sample_check.xlsx
│   │   └── final_validation_report.csv
│   │
│   ├── modeling/           # Output notebook 01b -- corpus siap SBERT/BERTopic
│   │   ├── berita_modeling.csv
│   │   └── ulasan_modeling.csv
│   │
│   └── processed/          # Output notebook 02 (SBERT + BERTopic)
│       ├── embeddings_berita.npy / embeddings_ulasan.npy
│       ├── embedding_index_berita.csv / embedding_index_ulasan.csv
│       ├── baseline_results.csv
│       ├── topic_model_results.csv
│       ├── topic_summary_berita.csv / topic_summary_ulasan.csv
│       ├── topic_temporal_berita.csv / topic_temporal_ulasan.csv
│       ├── document_topic_berita.csv / document_topic_ulasan.csv
│       ├── final_model_selection.csv
│       ├── model_config.json
│       ├── document_topic_berita.csv / document_topic_ulasan.csv   # (final, ditimpa notebook 02b)
│       ├── topic_summary_berita_final.csv / topic_summary_ulasan_final.csv
│       ├── topic_temporal_berita_final.csv / topic_temporal_ulasan_final.csv
│       ├── topic_trends_final.csv
│       ├── dashboard_topic_data.csv        # <- dibaca langsung oleh Streamlit
│       ├── final_topic_config.json
│       └── figures/        # Visualisasi (.html interaktif dari notebook 02 + .png dari 02b)
│
├── notebooks/
│   ├── 01_data_audit_cleaning.ipynb        # Audit data + cleaning
│   ├── 01b_relevance_validation.ipynb      # Validasi relevansi + corpus modeling
│   ├── 02_sbert_bertopic.ipynb             # SBERT + BERTopic + HDBSCAN + evaluasi
│   ├── 02b_topic_refinement_final.ipynb    # Perbaikan representasi topik + data final Streamlit (tahap sekarang)
│   └── 03_evaluation.ipynb                 # Evaluasi topik & kualitas cluster lebih lanjut (opsional)
│
├── app/
│   └── streamlit_app.py               # Dashboard (dikembangkan setelah topic modeling)
│
├── requirements.txt
└── README.md
```

## Cara Menjalankan (Google Colab)

1. Buka `notebooks/01_data_audit_cleaning.ipynb` di Google Colab.
2. Mount Google Drive, lalu letakkan kedua CSV mentah di `data/raw/`
   (path default di notebook: `/content/drive/MyDrive/project/data/raw/`,
   sesuaikan jika struktur Drive Anda berbeda).
3. Jalankan sel secara berurutan dari atas ke bawah:
   - **Tahap 1** — Audit kedua dataset (tanpa mengubah/menghapus apa pun).
   - **Tahap 2** — Ringkasan strategi cleaning (markdown, sudah disertakan
     alasan ilmiah tiap keputusan).
   - **Tahap 3** — Kode cleaning modular (`clean_news_text()`,
     `clean_review_text()`, `remove_duplicate_news()`, `validate_dates()`,
     `validate_reviews()`).
   - **Tahap 4** — Menyusun kolom dataset hasil sesuai skema yang disepakati.
   - **Tahap 5** — Flag kemungkinan relevansi (bukan filter final).
   - **Tahap 6** — Laporan kualitas SEBELUM vs SESUDAH (tabel metrik).
   - **Tahap 7** — Menyimpan dataset cleaned + audit report + log ke
     `data/cleaned/`.
4. Dataset mentah di `data/raw/` tidak pernah ditimpa — semua hasil cleaning
   masuk ke `data/cleaned/`.
5. Lanjutkan ke `notebooks/01b_relevance_validation.ipynb`:
   - Jalankan sel sampai Bagian 1-4 untuk menghasilkan file di `data/validation/`.
   - **Isi manual** `berita_manual_review.xlsx` (label `relevan`/`tidak_relevan` +
     alasan) dan periksa `berita_sample_check.xlsx` / `review_sample_check.xlsx`.
   - Jalankan sisa notebook (Bagian 7 dst.) untuk membentuk
     `data/modeling/berita_modeling.csv` dan `ulasan_modeling.csv` — dua corpus
     terpisah yang sudah divalidasi relevansinya, siap untuk `02_sbert_bertopic.ipynb`.
   - Notebook ini aman dijalankan ulang ("Run all"): file `berita_manual_review.xlsx`
     yang sudah diisi tidak akan ditimpa.
6. Lanjutkan ke `notebooks/02_sbert_bertopic.ipynb`:
   - Jalankan dari atas ke bawah. Baseline TF-IDF+KMeans, embedding SBERT (dengan cache —
     tidak dihitung ulang jika `.npy` sudah ada), lalu eksperimen HDBSCAN (4 nilai
     `min_cluster_size`) berjalan otomatis untuk Corpus A (berita) dan Corpus B (ulasan).
   - Periksa `final_model_selection.csv` untuk melihat konfigurasi yang direkomendasikan
     rubrik skor komposit, lalu **isi kolom `interpretasi_awal`** pada
     `topic_summary_berita.csv` / `topic_summary_ulasan.csv` setelah membaca top words +
     representative documents.
   - Lihat checklist lengkap di sel markdown terakhir notebook ini sebelum menulis Bab III/IV.
7. Lanjutkan ke `notebooks/02b_topic_refinement_final.ipynb` (tidak ada isian manual/Excel):
   - Membaca konfigurasi final (min_cluster_size) otomatis dari `final_model_selection.csv`,
     memuat ulang embedding dari cache, dan membangun ulang model BERTopic **sekali** per
     corpus (tidak sweep konfigurasi lagi).
   - Memperbaiki kata representasi topik dengan stopword Bahasa Indonesia + bigram (teks
     input SBERT tidak diubah), membuat `topic_label_auto` otomatis, lalu menghasilkan seluruh
     dataset final termasuk `dashboard_topic_data.csv` yang langsung dipakai Streamlit.
   - Menjalankan pengecekan konsistensi otomatis (assert) dan mencetak checklist akhir.

## Status Saat Ini

- [x] Scraping berita kesehatan Bangkalan (~10 tahun)
- [x] Scraping ulasan Google Maps kesehatan/lingkungan Bangkalan
- [x] Audit & cleaning dataset (notebook 01)
- [x] Validasi relevansi & pembentukan corpus modeling (notebook 01b)
- [x] SBERT embedding + BERTopic + HDBSCAN + evaluasi + analisis temporal (notebook 02)
- [x] Perbaikan representasi topik otomatis + data final siap Streamlit (notebook 02b) —
      **fokus README ini**
- [ ] Evaluasi topik lebih lanjut (notebook 03, opsional)
- [ ] Dashboard Streamlit (app/streamlit_app.py) — tinggal membaca `dashboard_topic_data.csv`

## Privasi

Kolom `reviewer_name` pada dataset ulasan **tidak digunakan sebagai fitur**
dan **tidak ditampilkan** di dataset cleaned maupun di dashboard. Kolom ini
hanya boleh disimpan pada dataset raw yang tidak dipublikasikan.
