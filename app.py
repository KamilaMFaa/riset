"""
Dashboard Identifikasi dan Analisis Tren Isu Kesehatan Masyarakat di Bangkalan
Menggunakan SBERT-BERTopic

Dashboard ini HANYA membaca hasil preprocessing & topic modeling yang sudah selesai
(notebook 01, 01b, 02, 02b). Tidak ada proses SBERT/embedding/BERTopic/HDBSCAN/UMAP
yang dijalankan di sini -- silakan lihat komentar di tiap fungsi load_*.

Perubahan pada versi ini (lihat CHANGELOG.md / penjelasan yang menyertai file ini):
- Menambahkan `topic_label_display`: label topic yang lebih ramah pengguna, dibuat
  dari `topic_label_auto` HANYA untuk tampilan (kata "Bangkalan" dihapus dari LABEL
  saja, lewat fungsi create_topic_display_label()). topic_label_auto, teks asli,
  top_words, dan document-topic assignment TIDAK diubah sama sekali.
- Halaman "Tren Isu" dipecah menjadi dua tab terpisah (Tren Berita & Tren Ulasan)
  dengan filter yang benar-benar spesifik untuk metadata masing-masing sumber, dan
  grafik yang dihitung ulang (jumlah dokumen & proporsi) berdasarkan agregasi
  langsung dari data/processed/dashboard_topic_data.csv sesuai filter yang aktif --
  BUKAN menjalankan ulang topic modeling, hanya pandas groupby dari hasil yang sudah
  ada, karena file agregat lama (topic_temporal_*_final.csv, topic_trends_final.csv)
  tidak memiliki granularitas kecamatan/tempat sehingga tidak bisa dipakai untuk
  tren yang responsif terhadap filter wilayah/tempat.
"""

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# KONFIGURASI HALAMAN & PATH DATA
# =============================================================================
st.set_page_config(
    page_title="Dashboard Isu Kesehatan Bangkalan",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path RELATIF terhadap root repository (bukan path absolut Google Colab).
# Jalankan aplikasi ini dari root repo, mis.: `streamlit run app.py`
DATA_DIR = Path("data/processed")

REQUIRED_FILES = {
    "dashboard_topic_data": DATA_DIR / "dashboard_topic_data.csv",
    "topic_summary_berita": DATA_DIR / "topic_summary_berita_final.csv",
    "topic_summary_ulasan": DATA_DIR / "topic_summary_ulasan_final.csv",
    "topic_temporal_berita": DATA_DIR / "topic_temporal_berita_final.csv",
    "topic_temporal_ulasan": DATA_DIR / "topic_temporal_ulasan_final.csv",
    "topic_trends": DATA_DIR / "topic_trends_final.csv",
    "final_topic_config": DATA_DIR / "final_topic_config.json",
}

OUTLIER_TOPIC_ID = -1
OUTLIER_LABEL = "Outlier / Tidak Terklasifikasi"

DISCLAIMER_LABEL = (
    "Topic yang ditampilkan merupakan hasil clustering teks menggunakan SBERT-BERTopic. "
    "Label topic (`topic_label_auto`) digunakan untuk membantu interpretasi dan **bukan** "
    "merupakan label ground truth. Label yang ditampilkan di UI (`topic_label_display`) "
    "hanyalah versi yang lebih ramah baca dari `topic_label_auto`."
)
DISCLAIMER_OUTLIER = (
    f"Topic {OUTLIER_TOPIC_ID} ({OUTLIER_LABEL}) merupakan dokumen yang tidak masuk ke topic "
    "reguler manapun -- bukan berarti dokumen tersebut membahas topik kesehatan tertentu."
)
DISCLAIMER_KETERBATASAN = (
    "Grafik pada dashboard ini menunjukkan **jumlah dokumen yang dianalisis** dan "
    "**proporsi dokumen** pada dataset penelitian (berita & ulasan Google Maps), "
    "**bukan** klaim tentang tingkat kesehatan masyarakat Bangkalan secara umum."
)


# =============================================================================
# PENGECEKAN FILE & LOADING DATA (CACHED)
# =============================================================================
def cek_file_tersedia() -> list:
    """Mengembalikan daftar file wajib yang TIDAK ditemukan (list path string)."""
    return [str(path) for path in REQUIRED_FILES.values() if not path.exists()]


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def beri_label_outlier(df: pd.DataFrame, kolom_topic: str = "topic",
                        kolom_label: str = "topic_label_auto") -> pd.DataFrame:
    """Memastikan topic -1 SELALU tampil sebagai OUTLIER_LABEL, apa pun isi kolom label aslinya."""
    if kolom_topic not in df.columns or kolom_label not in df.columns:
        return df
    df = df.copy()
    df.loc[df[kolom_topic] == OUTLIER_TOPIC_ID, kolom_label] = OUTLIER_LABEL
    return df


# =============================================================================
# LABEL TOPIC YANG RAMAH PENGGUNA (topic_label_display)
# =============================================================================
_KATA_GEOGRAFIS = {"bangkalan", "kabupaten bangkalan"}
_NORMALISASI_ISTILAH = {"covid19": "COVID-19", "covid-19": "COVID-19", "covid": "COVID-19"}


def create_topic_display_label(label_auto, top_words: str = "") -> str:
    """
    Membuat `topic_label_display` dari `topic_label_auto`, HANYA untuk tampilan.

    ATURAN (lihat juga poin 6 pada instruksi):
    - topic_label_auto TIDAK PERNAH diubah di dataset asli; fungsi ini hanya
      mengembalikan string baru.
    - Label outlier dibiarkan apa adanya.
    - Kata geografis "Bangkalan" dihapus HANYA dari label (bukan dari text,
      top_words, publisher, atau kolom metadata lain).
    - Jika setelah "Bangkalan" dihapus label menjadi kosong / tidak natural,
      fallback ke kata utama pada `top_words` (bukan label bebas yang tidak
      didukung data topic tersebut).
    """
    if label_auto is None or (isinstance(label_auto, float) and pd.isna(label_auto)):
        return label_auto
    label = str(label_auto).strip()
    if label == OUTLIER_LABEL:
        return label

    # Pisahkan label berdasarkan kata sambung "dan" atau koma, mis.
    # "Bangkalan, Sampah dan Lingkungan" -> ["Bangkalan", "Sampah", "Lingkungan"]
    bagian = re.split(r"\s*,\s*|\s+dan\s+", label)
    bagian = [b.strip() for b in bagian if b.strip()]

    bagian_bersih = [b for b in bagian if b.lower() not in _KATA_GEOGRAFIS]
    bagian_final = [_NORMALISASI_ISTILAH.get(b.lower(), b) for b in bagian_bersih]
    label_baru = " dan ".join(bagian_final).strip()

    # Fallback: label kosong / hanya sisa kata geografis -> pakai top_words
    if not label_baru or label_baru.lower() in _KATA_GEOGRAFIS:
        kata_utama = [
            w.strip() for w in str(top_words or "").split(",")
            if w.strip() and w.strip().lower() not in _KATA_GEOGRAFIS
        ]
        if kata_utama:
            label_baru = " dan ".join(w.capitalize() for w in kata_utama[:2])
        else:
            label_baru = label  # fallback terakhir: label asli, tidak diubah

    return label_baru


def _tambahkan_topic_label_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["topic_label_display"] = summary_df.apply(
        lambda r: create_topic_display_label(r.get("topic_label_auto"), r.get("top_words", "")),
        axis=1,
    )
    return summary_df


# =============================================================================
# UTILITAS FILTER KECAMATAN (mendukung nilai multi-kecamatan "A; B; C")
# =============================================================================
def split_kecamatan(nilai) -> list:
    if pd.isna(nilai):
        return []
    return [k.strip() for k in str(nilai).split(";") if k.strip()]


def get_kecamatan_options(df: pd.DataFrame, kolom: str = "kecamatan") -> list:
    semua = set()
    if kolom in df.columns:
        for v in df[kolom].dropna():
            semua.update(split_kecamatan(v))
    return sorted(semua)


def match_kecamatan(nilai, kecamatan_target) -> bool:
    """True jika kecamatan_target ada pada daftar kecamatan baris ini (bisa multi-kecamatan)."""
    if kecamatan_target is None:
        return True
    return kecamatan_target in split_kecamatan(nilai)


# =============================================================================
# LOADING SELURUH DATA (CACHED, dipanggil sekali)
# =============================================================================
@st.cache_data
def load_semua_data() -> dict:
    """Memuat seluruh file data yang dibutuhkan dashboard. Dipanggil SEKALI (cached)."""
    data = {}
    data["dashboard"] = load_csv(str(REQUIRED_FILES["dashboard_topic_data"]))
    data["dashboard"] = beri_label_outlier(data["dashboard"])

    data["summary_berita"] = _tambahkan_topic_label_display(
        beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_summary_berita"])))
    )
    data["summary_ulasan"] = _tambahkan_topic_label_display(
        beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_summary_ulasan"])))
    )

    data["temporal_berita"] = load_csv(str(REQUIRED_FILES["topic_temporal_berita"]))
    data["temporal_ulasan"] = load_csv(str(REQUIRED_FILES["topic_temporal_ulasan"]))

    data["trends"] = beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_trends"])))

    data["config"] = load_json(str(REQUIRED_FILES["final_topic_config"]))

    # --- Bangun topic_label_display konsisten di dashboard, berdasarkan mapping ---
    # per (source, topic) yang sudah dihitung sekali di tabel summary masing-masing.
    map_berita = dict(zip(data["summary_berita"]["topic"], data["summary_berita"]["topic_label_display"]))
    map_ulasan = dict(zip(data["summary_ulasan"]["topic"], data["summary_ulasan"]["topic_label_display"]))

    df = data["dashboard"]
    df["topic_label_display"] = df["topic_label_auto"]
    mask_berita = df["source"] == "berita"
    mask_ulasan = df["source"] == "ulasan"
    df.loc[mask_berita, "topic_label_display"] = df.loc[mask_berita, "topic"].map(map_berita)
    df.loc[mask_ulasan, "topic_label_display"] = df.loc[mask_ulasan, "topic"].map(map_ulasan)
    data["dashboard"] = df

    return data


def potong_teks(teks: str, panjang: int = 200) -> str:
    teks = "" if pd.isna(teks) else str(teks)
    return teks if len(teks) <= panjang else teks[:panjang].rstrip() + "..."


# =============================================================================
# HALAMAN 1 -- BERANDA
# =============================================================================
def halaman_beranda(data: dict):
    st.title("Dashboard Identifikasi dan Analisis Tren Isu Kesehatan Masyarakat di Bangkalan")
    st.caption("Menggunakan SBERT-BERTopic -- media visualisasi hasil penelitian, bukan alat diagnosis kesehatan.")

    st.markdown(
        "Dashboard ini menyajikan hasil analisis topik dari dua sumber data digital: "
        "**berita lokal** dan **ulasan Google Maps** terkait kesehatan & lingkungan di "
        "Kabupaten Bangkalan. Topik ditemukan lewat *topic modeling* semantik (SBERT + "
        "BERTopic + HDBSCAN), lalu dianalisis tren kemunculannya dari waktu ke waktu."
    )
    st.info(DISCLAIMER_LABEL)

    df = data["dashboard"]
    df_berita = df[df["source"] == "berita"]
    df_ulasan = df[df["source"] == "ulasan"]

    jumlah_topic_berita = data["summary_berita"].loc[
        data["summary_berita"]["topic"] != OUTLIER_TOPIC_ID, "topic"
    ].nunique()
    jumlah_topic_ulasan = data["summary_ulasan"].loc[
        data["summary_ulasan"]["topic"] != OUTLIER_TOPIC_ID, "topic"
    ].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Dokumen", f"{len(df):,}".replace(",", "."))
    col2.metric("Total Berita", f"{len(df_berita):,}".replace(",", "."))
    col3.metric("Total Ulasan", f"{len(df_ulasan):,}".replace(",", "."))
    col4.metric("Topic Berita (Reguler)", jumlah_topic_berita)
    col5.metric("Topic Ulasan (Reguler)", jumlah_topic_ulasan)

    st.divider()

    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.subheader("Distribusi Sumber Data")
        distribusi_sumber = df["source"].value_counts().rename_axis("source").reset_index(name="jumlah")
        distribusi_sumber["source"] = distribusi_sumber["source"].replace(
            {"berita": "Berita", "ulasan": "Ulasan"}
        )
        fig = px.bar(distribusi_sumber, x="source", y="jumlah", text="jumlah",
                     labels={"source": "Sumber", "jumlah": "Jumlah Dokumen"})
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    with kanan:
        st.subheader("Konfigurasi Ringkas")
        cfg = data["config"]
        if "tahun_bulan" in df.columns and df["tahun_bulan"].notna().any():
            periode_awal = df["tahun_bulan"].min()
            periode_akhir = df["tahun_bulan"].max()
            st.write(f"**Periode data:** {periode_awal} s.d. {periode_akhir}")
        st.write(f"**Model SBERT:** `{cfg.get('sbert_model_used', '-')}`")
        st.write("**Metode topic modeling:** BERTopic (c-TF-IDF representation)")
        st.write(
            f"**HDBSCAN min_cluster_size:** berita = "
            f"{cfg.get('hdbscan', {}).get('min_cluster_size_berita', '-')}, ulasan = "
            f"{cfg.get('hdbscan', {}).get('min_cluster_size_ulasan', '-')}"
        )
        st.caption("Detail metodologi lengkap ada di halaman **Metodologi**.")

    st.caption(DISCLAIMER_KETERBATASAN)


# =============================================================================
# HALAMAN 2 -- ANALISIS TOPIC
# =============================================================================
def _gabungkan_summary(data: dict, pilihan_dataset: str) -> pd.DataFrame:
    """Menggabungkan topic_summary berita & ulasan sesuai pilihan dataset, dengan kolom 'dataset'."""
    s_berita = data["summary_berita"].copy()
    s_berita.insert(0, "dataset", "Berita")
    s_ulasan = data["summary_ulasan"].copy()
    s_ulasan.insert(0, "dataset", "Ulasan")

    if pilihan_dataset == "Berita":
        return s_berita
    elif pilihan_dataset == "Ulasan":
        return s_ulasan
    return pd.concat([s_berita, s_ulasan], ignore_index=True)


def halaman_analisis_topic(data: dict):
    st.title("Analisis Topic")
    st.info(DISCLAIMER_LABEL)

    pilihan_dataset = st.selectbox("Dataset", ["Semua", "Berita", "Ulasan"], index=0)
    ringkasan = _gabungkan_summary(data, pilihan_dataset)

    reguler = ringkasan[ringkasan["topic"] != OUTLIER_TOPIC_ID].sort_values("topic_size", ascending=False)
    outlier = ringkasan[ringkasan["topic"] == OUTLIER_TOPIC_ID]

    st.subheader("Distribusi Topic")
    if reguler.empty:
        st.warning("Tidak ada topik reguler pada pilihan dataset ini.")
    else:
        fig = px.bar(
            reguler, x="topic_label_display", y="topic_size", color="dataset" if pilihan_dataset == "Semua" else None,
            text="topic_size", barmode="group",
            labels={"topic_label_display": "Topic", "topic_size": "Jumlah Dokumen", "dataset": "Sumber"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Jumlah Dokumen, Persentase, dan Top Words per Topic")
    kolom_tampil = ["dataset", "topic", "topic_label_display", "topic_label_auto", "topic_size", "percentage", "top_words"]
    kolom_tampil = [k for k in kolom_tampil if k in ringkasan.columns]
    tabel_tampil = ringkasan[kolom_tampil].sort_values("topic_size", ascending=False)
    if "percentage" in tabel_tampil.columns:
        tabel_tampil = tabel_tampil.assign(percentage=(tabel_tampil["percentage"] * 100).round(1))
    st.dataframe(
        tabel_tampil, width="stretch", hide_index=True,
        column_config={
            "topic_label_display": st.column_config.TextColumn("Topic (label ramah pengguna)"),
            "topic_label_auto": st.column_config.TextColumn("topic_label_auto (teknis)"),
            "percentage": st.column_config.NumberColumn("Persentase (%)", format="%.1f%%"),
        },
    )

    if not outlier.empty:
        for _, row in outlier.iterrows():
            st.caption(
                f"**{row['dataset']}** -- {OUTLIER_LABEL}: {int(row['topic_size'])} dokumen "
                f"({row['percentage'] * 100:.1f}% dari dataset tsb.). {DISCLAIMER_OUTLIER}"
            )

    st.divider()
    st.subheader("Detail Topic")
    if reguler.empty:
        st.caption("Tidak ada topik reguler untuk ditampilkan detailnya.")
    else:
        opsi_topic = reguler.apply(
            lambda r: f"[{r['dataset']}] {r['topic_label_display']} (Topic {r['topic']})", axis=1
        ).tolist()
        pilihan = st.selectbox("Pilih topic", opsi_topic)
        idx_dipilih = opsi_topic.index(pilihan)
        baris_topic = reguler.iloc[idx_dipilih]

        dataset_key = "berita" if baris_topic["dataset"] == "Berita" else "ulasan"
        topic_id = baris_topic["topic"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Dokumen", int(baris_topic["topic_size"]))
        c2.metric("Persentase", f"{baris_topic['percentage'] * 100:.1f}%")
        c3.metric("Topic ID", int(topic_id))

        st.write("**Top words:**", baris_topic["top_words"])
        st.caption(f"Label teknis asli (`topic_label_auto`): {baris_topic['topic_label_auto']}")
        if "representative_document_ids" in baris_topic and pd.notna(baris_topic["representative_document_ids"]):
            st.write("**Contoh document_id representatif:**", baris_topic["representative_document_ids"])

        df_dok_topic = data["dashboard"][
            (data["dashboard"]["source"] == dataset_key) & (data["dashboard"]["topic"] == topic_id)
        ]
        with st.expander(f"Lihat contoh dokumen pada topic ini (menampilkan hingga 5 dari {len(df_dok_topic)})"):
            for _, dok in df_dok_topic.head(5).iterrows():
                if dataset_key == "berita":
                    st.markdown(
                        f"- **Publisher:** {dok.get('publisher', '-')} | "
                        f"**Kecamatan:** {dok.get('kecamatan', '-')} | "
                        f"**Periode:** {dok.get('tahun_bulan', '-')}"
                        + (f" | [Buka URL]({dok['url']})" if pd.notna(dok.get("url")) else "")
                    )
                else:
                    st.markdown(
                        f"- **Tempat:** {dok.get('place_name', '-')} | "
                        f"**Kategori:** {dok.get('place_category', '-')} | "
                        f"**Rating:** {dok.get('rating', '-')} | "
                        f"**Periode:** {dok.get('tahun_bulan', '-')}"
                    )
                st.caption(potong_teks(dok.get("text", ""), 220))

    # --- Analisis Wilayah (khusus berita, jika kolom kecamatan tersedia) ---
    if pilihan_dataset in ("Semua", "Berita") and "kecamatan" in data["dashboard"].columns:
        st.divider()
        st.subheader("Analisis Wilayah (Berita)")
        df_berita_all = data["dashboard"][data["dashboard"]["source"] == "berita"]
        if df_berita_all["kecamatan"].notna().any():
            per_kecamatan = (
                df_berita_all.groupby("kecamatan").size().rename("jumlah_dokumen")
                .reset_index().sort_values("jumlah_dokumen", ascending=False)
            )
            fig_wilayah = px.bar(per_kecamatan, x="kecamatan", y="jumlah_dokumen",
                                  labels={"kecamatan": "Kecamatan", "jumlah_dokumen": "Distribusi Dokumen yang Dianalisis"})
            st.plotly_chart(fig_wilayah, width="stretch")

            topic_dominan = (
                df_berita_all[df_berita_all["topic"] != OUTLIER_TOPIC_ID]
                .groupby("kecamatan")["topic_label_display"]
                .agg(lambda s: s.value_counts().idxmax() if len(s) else "-")
                .rename("topic_paling_sering_muncul").reset_index()
            )
            st.caption(
                "Topic yang paling sering muncul per kecamatan (deskriptif, berdasarkan dokumen "
                "yang dianalisis -- **bukan** klaim tingkat kesehatan masyarakat kecamatan tsb.):"
            )
            st.dataframe(topic_dominan, width="stretch", hide_index=True)
        else:
            st.caption("Kolom kecamatan tidak memiliki data yang cukup untuk ditampilkan.")

    # --- Analisis Ulasan (2 topic reguler: distribusi rating per topic) ---
    if pilihan_dataset in ("Semua", "Ulasan"):
        st.divider()
        st.subheader("Analisis Ulasan: Distribusi Rating per Topic")
        st.caption(DISCLAIMER_LABEL)
        df_ulasan_all = data["dashboard"][data["dashboard"]["source"] == "ulasan"].copy()
        df_ulasan_all = df_ulasan_all[df_ulasan_all["topic"] != OUTLIER_TOPIC_ID]
        if df_ulasan_all.empty or "rating" not in df_ulasan_all.columns:
            st.caption("Tidak ada data rating pada topik reguler ulasan.")
        else:
            fig_rating = px.histogram(
                df_ulasan_all, x="rating", color="topic_label_display", barmode="group",
                labels={"rating": "Rating", "topic_label_display": "Topic"},
            )
            fig_rating.update_xaxes(dtick=1)
            st.plotly_chart(fig_rating, width="stretch")


# =============================================================================
# HALAMAN 3 -- TREN ISU (Tren Berita & Tren Ulasan, terpisah)
# =============================================================================
def _hitung_tren(df_denom: pd.DataFrame, topic_col: str, topics_dipilih: list) -> pd.DataFrame:
    """
    Menghitung jumlah dokumen & proporsi per topic per periode (tahun_bulan).

    df_denom HARUS SUDAH difilter sesuai wilayah/tempat & periode yang sedang aktif.
    Proporsi = jumlah dokumen topic / total dokumen pada periode & filter aktif
    (denominator = seluruh dokumen pada df_denom untuk periode tsb, termasuk topic
    lain, supaya proporsi selalu dihitung ulang sesuai filter wilayah/tempat -- BUKAN
    proporsi global).
    """
    kolom_hasil = ["tahun_bulan", topic_col, "jumlah_dokumen", "total_dokumen", "proporsi", "tahun_bulan_dt"]
    if df_denom.empty or not topics_dipilih:
        return pd.DataFrame(columns=kolom_hasil)

    total_periode = df_denom.groupby("tahun_bulan").size().rename("total_dokumen")

    df_topic = df_denom[df_denom[topic_col].isin(topics_dipilih)]
    if df_topic.empty:
        return pd.DataFrame(columns=kolom_hasil)

    hasil = df_topic.groupby(["tahun_bulan", topic_col]).size().rename("jumlah_dokumen").reset_index()
    hasil = hasil.merge(total_periode, on="tahun_bulan", how="left")
    hasil["proporsi"] = hasil["jumlah_dokumen"] / hasil["total_dokumen"]
    hasil["tahun_bulan_dt"] = pd.to_datetime(hasil["tahun_bulan"], format="%Y-%m", errors="coerce")
    hasil = hasil.sort_values("tahun_bulan_dt")
    return hasil


def _plot_tren(hasil: pd.DataFrame, topic_col: str, kolom_y: str, label_y: str, key: str):
    fig = px.line(
        hasil, x="tahun_bulan_dt", y=kolom_y, color=topic_col, markers=True,
        labels={"tahun_bulan_dt": "Periode", kolom_y: label_y, topic_col: "Topic"},
    )
    fig.update_xaxes(tickformat="%Y-%m", tickangle=-45)
    st.plotly_chart(fig, width="stretch", key=key)

    with st.expander("Lihat data tren dalam tabel"):
        tampil = hasil[["tahun_bulan", topic_col, "jumlah_dokumen", "total_dokumen", "proporsi"]].rename(
            columns={"tahun_bulan": "Periode", topic_col: "Topic", "total_dokumen": "Total Dokumen (periode & filter aktif)"}
        )
        st.dataframe(tampil, width="stretch", hide_index=True)


def _judul_tren_berita(kecamatan_dipilih: str, topics_dipilih: list) -> str:
    lokasi = "Seluruh Kabupaten Bangkalan" if kecamatan_dipilih == "Semua Kecamatan" else f"Kecamatan {kecamatan_dipilih}"
    if len(topics_dipilih) == 1:
        return f"Tren Topik {topics_dipilih[0]} di {lokasi}"
    return f"Tren Isu ({', '.join(topics_dipilih)}) di {lokasi}"


def _tren_berita(data: dict):
    df_b = data["dashboard"][data["dashboard"]["source"] == "berita"].copy()

    topic_opsi = sorted(df_b.loc[df_b["topic"] != OUTLIER_TOPIC_ID, "topic_label_display"].dropna().unique().tolist())
    kec_opsi = get_kecamatan_options(df_b, "kecamatan")

    col1, col2 = st.columns([1, 1])
    with col1:
        kecamatan_dipilih = st.selectbox("Kecamatan", ["Semua Kecamatan"] + kec_opsi, key="tren_berita_kec")
    with col2:
        metrik = st.radio("Metrik", ["Jumlah Dokumen", "Proporsi"], horizontal=True, key="tren_berita_metrik")

    topics_dipilih = st.multiselect("Topic", topic_opsi, default=topic_opsi, key="tren_berita_topic")

    tahun_tersedia = sorted(df_b["tahun"].dropna().unique().tolist())
    if len(tahun_tersedia) >= 2:
        rentang_tahun = st.slider(
            "Periode (rentang tahun)", int(min(tahun_tersedia)), int(max(tahun_tersedia)),
            (int(min(tahun_tersedia)), int(max(tahun_tersedia))), key="tren_berita_tahun",
        )
    else:
        t = int(tahun_tersedia[0]) if tahun_tersedia else 0
        rentang_tahun = (t, t)

    if not topics_dipilih:
        st.warning("Pilih minimal satu topic untuk menampilkan tren.")
        return

    kec_target = None if kecamatan_dipilih == "Semua Kecamatan" else kecamatan_dipilih
    mask_kec = df_b["kecamatan"].apply(lambda v: match_kecamatan(v, kec_target))
    mask_tahun = df_b["tahun"].between(rentang_tahun[0], rentang_tahun[1])
    df_denom = df_b[mask_kec & mask_tahun]

    if df_denom.empty:
        st.warning("Tidak ada data pada kombinasi filter ini.")
        return

    hasil = _hitung_tren(df_denom, "topic_label_display", topics_dipilih)
    if hasil.empty:
        st.warning("Tidak ada dokumen dengan topic terpilih pada kombinasi filter ini.")
        return

    kolom_y = "jumlah_dokumen" if metrik == "Jumlah Dokumen" else "proporsi"
    label_y = "Jumlah Dokumen" if metrik == "Jumlah Dokumen" else "Proporsi Dokumen"

    st.markdown(f"#### {_judul_tren_berita(kecamatan_dipilih, topics_dipilih)}")
    _plot_tren(hasil, "topic_label_display", kolom_y, label_y, key="chart_tren_berita")


def _judul_tren_ulasan(tempat_dipilih: str, topics_dipilih: list) -> str:
    lokasi = "Seluruh Tempat yang Dianalisis" if tempat_dipilih == "Semua Tempat" else tempat_dipilih
    if len(topics_dipilih) == 1:
        return f"Tren Topic {topics_dipilih[0]} pada {lokasi}"
    return f"Tren Topic ({', '.join(topics_dipilih)}) pada {lokasi}"


def _tren_ulasan(data: dict):
    df_u = data["dashboard"][data["dashboard"]["source"] == "ulasan"].copy()

    topic_opsi = sorted(df_u.loc[df_u["topic"] != OUTLIER_TOPIC_ID, "topic_label_display"].dropna().unique().tolist())
    kategori_opsi = sorted(df_u["place_category"].dropna().unique().tolist())

    col1, col2 = st.columns([1, 1])
    with col1:
        kategori_dipilih = st.multiselect("Kategori Tempat", kategori_opsi, default=kategori_opsi, key="tren_ulasan_kategori")

    df_u_kategori = df_u[df_u["place_category"].isin(kategori_dipilih)] if kategori_dipilih else df_u.iloc[0:0]
    tempat_opsi = sorted(df_u_kategori["place_name"].dropna().unique().tolist())

    with col2:
        tempat_dipilih = st.selectbox("Tempat", ["Semua Tempat"] + tempat_opsi, key="tren_ulasan_tempat")

    metrik = st.radio("Metrik", ["Jumlah Dokumen", "Proporsi"], horizontal=True, key="tren_ulasan_metrik")
    topics_dipilih = st.multiselect("Topic", topic_opsi, default=topic_opsi, key="tren_ulasan_topic")

    tahun_tersedia = sorted(df_u["tahun"].dropna().unique().tolist())
    if len(tahun_tersedia) >= 2:
        rentang_tahun = st.slider(
            "Periode (rentang tahun)", int(min(tahun_tersedia)), int(max(tahun_tersedia)),
            (int(min(tahun_tersedia)), int(max(tahun_tersedia))), key="tren_ulasan_tahun",
        )
    else:
        t = int(tahun_tersedia[0]) if tahun_tersedia else 0
        rentang_tahun = (t, t)

    if not topics_dipilih:
        st.warning("Pilih minimal satu topic untuk menampilkan tren.")
        return
    if not kategori_dipilih:
        st.warning("Pilih minimal satu kategori tempat.")
        return

    mask_kategori = df_u["place_category"].isin(kategori_dipilih)
    mask_tempat = pd.Series(True, index=df_u.index) if tempat_dipilih == "Semua Tempat" else (df_u["place_name"] == tempat_dipilih)
    mask_tahun = df_u["tahun"].between(rentang_tahun[0], rentang_tahun[1])
    df_denom = df_u[mask_kategori & mask_tempat & mask_tahun]

    if df_denom.empty:
        st.warning("Tidak ada data pada kombinasi filter ini.")
        return

    hasil = _hitung_tren(df_denom, "topic_label_display", topics_dipilih)
    if hasil.empty:
        st.warning("Tidak ada dokumen dengan topic terpilih pada kombinasi filter ini.")
        return

    kolom_y = "jumlah_dokumen" if metrik == "Jumlah Dokumen" else "proporsi"
    label_y = "Jumlah Dokumen" if metrik == "Jumlah Dokumen" else "Proporsi Dokumen"

    st.markdown(f"#### {_judul_tren_ulasan(tempat_dipilih, topics_dipilih)}")
    _plot_tren(hasil, "topic_label_display", kolom_y, label_y, key="chart_tren_ulasan")


def halaman_tren_isu(data: dict):
    st.title("Tren Isu dari Waktu ke Waktu")
    st.info(DISCLAIMER_LABEL)
    st.caption(
        "Filter Tren Berita dan Tren Ulasan dipisah karena metadata kedua sumber berbeda "
        "(berita: kecamatan/publisher, ulasan: tempat/kategori tempat/rating)."
    )
    tab_berita, tab_ulasan = st.tabs(["📰 Tren Berita", "⭐ Tren Ulasan"])
    with tab_berita:
        _tren_berita(data)
    with tab_ulasan:
        _tren_ulasan(data)

    st.divider()
    st.caption(DISCLAIMER_KETERBATASAN)


# =============================================================================
# HALAMAN 4 -- EKSPLORASI DATA
# =============================================================================
def halaman_eksplorasi_data(data: dict):
    st.title("Eksplorasi Data")
    st.caption("Menjelajahi dokumen individual beserta topic hasil analisis. Data identitas pribadi tidak ditampilkan.")

    df = data["dashboard"].copy()

    with st.sidebar:
        st.markdown("### Filter Eksplorasi Data")
        pilihan_sumber = st.multiselect("Sumber", ["berita", "ulasan"], default=["berita", "ulasan"])
        df_f = df[df["source"].isin(pilihan_sumber)]

        if "tahun" in df_f.columns and df_f["tahun"].notna().any():
            tahun_opsi = sorted(df_f["tahun"].dropna().unique().tolist())
            tahun_dipilih = st.multiselect("Tahun", tahun_opsi, default=tahun_opsi)
            df_f = df_f[df_f["tahun"].isin(tahun_dipilih)]

        if "topic_label_display" in df_f.columns:
            topic_opsi = sorted(df_f["topic_label_display"].dropna().unique().tolist())
            topic_dipilih = st.multiselect("Topic", topic_opsi, default=topic_opsi)
            df_f = df_f[df_f["topic_label_display"].isin(topic_dipilih)]

        if "kecamatan" in df_f.columns and df_f["kecamatan"].notna().any():
            kec_opsi = get_kecamatan_options(df_f, "kecamatan")
            kec_dipilih = st.multiselect("Kecamatan (berita)", kec_opsi, default=kec_opsi)
            if kec_dipilih:
                mask_kec = df_f["kecamatan"].apply(
                    lambda v: any(k in kec_dipilih for k in split_kecamatan(v))
                )
                df_f = df_f[mask_kec | df_f["kecamatan"].isna()]

        if "publisher" in df_f.columns and df_f["publisher"].notna().any():
            pub_opsi = sorted(df_f["publisher"].dropna().unique().tolist())
            pub_dipilih = st.multiselect("Publisher (berita)", pub_opsi, default=pub_opsi)
            df_f = df_f[df_f["publisher"].isin(pub_dipilih) | df_f["publisher"].isna()]

        if "place_name" in df_f.columns and df_f["place_name"].notna().any():
            tempat_opsi = sorted(df_f["place_name"].dropna().unique().tolist())
            tempat_dipilih = st.multiselect("Tempat (ulasan)", tempat_opsi, default=tempat_opsi)
            df_f = df_f[df_f["place_name"].isin(tempat_dipilih) | df_f["place_name"].isna()]

        if "place_category" in df_f.columns and df_f["place_category"].notna().any():
            kat_opsi = sorted(df_f["place_category"].dropna().unique().tolist())
            kat_dipilih = st.multiselect("Kategori Tempat (ulasan)", kat_opsi, default=kat_opsi)
            df_f = df_f[df_f["place_category"].isin(kat_dipilih) | df_f["place_category"].isna()]

        if "rating" in df_f.columns and df_f["rating"].notna().any():
            rating_min = int(df_f["rating"].min())
            rating_max = int(df_f["rating"].max())
            if rating_min < rating_max:
                rentang_rating = st.slider("Rating (ulasan)", rating_min, rating_max, (rating_min, rating_max))
                df_f = df_f[df_f["rating"].between(*rentang_rating) | df_f["rating"].isna()]

    st.write(f"**Jumlah baris hasil filter:** {len(df_f):,}".replace(",", "."))

    # Tidak menampilkan reviewer_name / alamat lengkap -- kolom tsb memang tidak ada di
    # dashboard_topic_data.csv (sudah dihilangkan sejak notebook 01b), dipastikan lagi di sini.
    kolom_terlarang = ["reviewer_name", "place_address"]
    kolom_tampil = [k for k in df_f.columns if k not in kolom_terlarang]

    df_tampil = df_f[kolom_tampil].copy()
    df_tampil["text"] = df_tampil["text"].apply(lambda t: potong_teks(t, 180))

    column_config = {
        "topic_label_display": st.column_config.TextColumn("Topic (label ramah pengguna)"),
        "topic_label_auto": st.column_config.TextColumn("topic_label_auto (teknis)"),
    }
    if "url" in df_tampil.columns:
        column_config["url"] = st.column_config.LinkColumn("URL")

    st.dataframe(df_tampil, width="stretch", hide_index=True, column_config=column_config)


# =============================================================================
# HALAMAN 5 -- METODOLOGI
# =============================================================================
def halaman_metodologi(data: dict):
    st.title("Metodologi")
    cfg = data["config"]

    st.markdown("""
    **Alur pipeline penelitian:**

    ```
    Data digital
      -> Cleaning
      -> Validasi relevansi
      -> SBERT (embedding semantik)
      -> UMAP (reduksi dimensi)
      -> HDBSCAN (clustering)
      -> BERTopic (representasi topic)
      -> Analisis temporal
      -> Dashboard (halaman ini)
    ```
    """)
    st.info(DISCLAIMER_LABEL + " " + DISCLAIMER_OUTLIER)

    st.subheader("Konfigurasi Model")
    umap_cfg = cfg.get("umap_config", {})
    hdbscan_cfg = cfg.get("hdbscan", {})
    hasil = cfg.get("hasil", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SBERT**")
        st.write(f"- Model: `{cfg.get('sbert_model_used', '-')}`")
        st.write(f"- Random state: `{cfg.get('random_state', '-')}`")

        st.markdown("**UMAP**")
        for k, v in umap_cfg.items():
            st.write(f"- {k}: `{v}`")

    with c2:
        st.markdown("**HDBSCAN**")
        st.write(f"- min_cluster_size (berita): `{hdbscan_cfg.get('min_cluster_size_berita', '-')}`")
        st.write(f"- min_cluster_size (ulasan): `{hdbscan_cfg.get('min_cluster_size_ulasan', '-')}`")
        st.write(f"- metric: `{hdbscan_cfg.get('metric', '-')}`")
        st.write(f"- cluster_selection_method: `{hdbscan_cfg.get('cluster_selection_method', '-')}`")

        st.markdown("**Representasi Topic**")
        rep_cfg = cfg.get("representasi_topik", {})
        st.write(f"- Vectorizer: `{rep_cfg.get('vectorizer', '-')}`, ngram_range: "
                 f"`{rep_cfg.get('ngram_range', '-')}`, min_df: `{rep_cfg.get('min_df', '-')}`")
        st.write(f"- Stopword: {rep_cfg.get('stop_words', '-')}")

    st.divider()
    st.subheader("Ringkasan Hasil Topic Modeling")
    for nama, label in [("berita", "Berita"), ("ulasan", "Ulasan")]:
        h = hasil.get(nama, {})
        if not h:
            continue
        st.markdown(f"**{label}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Dokumen", h.get("total_documents", "-"))
        c2.metric("Jumlah Topic Reguler", h.get("total_regular_topics", "-"))
        c3.metric("Jumlah Outlier", h.get("total_outliers", "-"))
        c4.metric("Persentase Outlier", f"{h.get('outlier_percentage', 0) * 100:.1f}%")

    st.divider()
    st.markdown("**Label topic pada dashboard (`topic_label_display`)**")
    st.write(
        "Kolom `topic_label_display` dibuat dari `topic_label_auto` khusus untuk tampilan "
        "(fungsi `create_topic_display_label()`), dengan menghapus kata geografis \"Bangkalan\" "
        "dari label agar lebih natural dibaca. Label teknis asli (`topic_label_auto`), teks "
        "dokumen, top_words, dan document-topic assignment TIDAK diubah."
    )

    st.divider()
    st.caption(
        "Seluruh proses SBERT, embedding, UMAP, HDBSCAN, dan BERTopic dijalankan sebelumnya di "
        "Google Colab (notebook 01, 01b, 02, 02b) -- dashboard ini hanya membaca hasil akhirnya."
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    file_hilang = cek_file_tersedia()
    if file_hilang:
        st.error(
            "Dashboard tidak dapat dijalankan karena file data berikut tidak ditemukan:\n\n"
            + "\n".join(f"- `{f}`" for f in file_hilang)
            + "\n\nPastikan folder `data/processed/` berisi seluruh hasil dari "
              "`02b_topic_refinement_final.ipynb`, dan aplikasi dijalankan dari root repository."
        )
        st.stop()

    data = load_semua_data()

    st.sidebar.title("🩺 Navigasi")
    halaman = st.sidebar.radio(
        "Pilih halaman",
        ["Beranda", "Analisis Topic", "Tren Isu", "Eksplorasi Data", "Metodologi"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Dashboard ini hanya membaca hasil preprocessing & topic modeling yang sudah "
        "selesai. Tidak ada proses SBERT/BERTopic yang dijalankan di sini."
    )

    if halaman == "Beranda":
        halaman_beranda(data)
    elif halaman == "Analisis Topic":
        halaman_analisis_topic(data)
    elif halaman == "Tren Isu":
        halaman_tren_isu(data)
    elif halaman == "Eksplorasi Data":
        halaman_eksplorasi_data(data)
    elif halaman == "Metodologi":
        halaman_metodologi(data)


if __name__ == "__main__":
    main()