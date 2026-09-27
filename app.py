"""
Dashboard Identifikasi dan Analisis Tren Isu Kesehatan Masyarakat di Bangkalan
Menggunakan SBERT-BERTopic

Dashboard ini HANYA membaca hasil preprocessing & topic modeling yang sudah selesai
(notebook 01, 01b, 02, 02b). Tidak ada proses SBERT/embedding/BERTopic/HDBSCAN/UMAP
yang dijalankan di sini -- silakan lihat komentar di tiap fungsi load_*.
"""

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
    "merupakan label ground truth."
)
DISCLAIMER_OUTLIER = (
    f"Topic {OUTLIER_TOPIC_ID} ({OUTLIER_LABEL}) merupakan dokumen yang tidak masuk ke topic "
    "reguler manapun -- bukan berarti dokumen tersebut membahas topik kesehatan tertentu."
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


@st.cache_data
def load_semua_data() -> dict:
    """Memuat seluruh file data yang dibutuhkan dashboard. Dipanggil SEKALI (cached)."""
    data = {}
    data["dashboard"] = load_csv(str(REQUIRED_FILES["dashboard_topic_data"]))
    data["dashboard"] = beri_label_outlier(data["dashboard"])

    data["summary_berita"] = beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_summary_berita"])))
    data["summary_ulasan"] = beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_summary_ulasan"])))

    data["temporal_berita"] = load_csv(str(REQUIRED_FILES["topic_temporal_berita"]))
    data["temporal_ulasan"] = load_csv(str(REQUIRED_FILES["topic_temporal_ulasan"]))

    data["trends"] = beri_label_outlier(load_csv(str(REQUIRED_FILES["topic_trends"])))

    data["config"] = load_json(str(REQUIRED_FILES["final_topic_config"]))
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
            reguler, x="topic_label_auto", y="topic_size", color="dataset" if pilihan_dataset == "Semua" else None,
            text="topic_size", barmode="group",
            labels={"topic_label_auto": "Topic", "topic_size": "Jumlah Dokumen", "dataset": "Sumber"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Jumlah Dokumen, Persentase, dan Top Words per Topic")
    kolom_tampil = ["dataset", "topic", "topic_label_auto", "topic_size", "percentage", "top_words"]
    kolom_tampil = [k for k in kolom_tampil if k in ringkasan.columns]
    tabel_tampil = ringkasan[kolom_tampil].sort_values("topic_size", ascending=False)
    if "percentage" in tabel_tampil.columns:
        tabel_tampil = tabel_tampil.assign(percentage=(tabel_tampil["percentage"] * 100).round(1))
    st.dataframe(tabel_tampil, width="stretch", hide_index=True,
                 column_config={"percentage": st.column_config.NumberColumn("Persentase (%)", format="%.1f%%")})

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
            lambda r: f"[{r['dataset']}] Topic {r['topic']} -- {r['topic_label_auto']}", axis=1
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
                .groupby("kecamatan")["topic_label_auto"]
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
                df_ulasan_all, x="rating", color="topic_label_auto", barmode="group",
                labels={"rating": "Rating", "topic_label_auto": "Topic"},
            )
            fig_rating.update_xaxes(dtick=1)
            st.plotly_chart(fig_rating, width="stretch")


# =============================================================================
# HALAMAN 3 -- TREN ISU
# =============================================================================
def halaman_tren_isu(data: dict):
    st.title("Tren Isu dari Waktu ke Waktu")
    st.info(DISCLAIMER_LABEL)

    trends = data["trends"].copy()
    trends["dataset"] = trends["dataset"].replace({"berita": "Berita", "ulasan": "Ulasan"})

    pilihan_dataset = st.radio("Dataset", ["Berita", "Ulasan"], horizontal=True)
    subset = trends[trends["dataset"] == pilihan_dataset]
    subset = subset[subset["topic"] != OUTLIER_TOPIC_ID]  # tren outlier tidak substantif untuk dianalisis

    if subset.empty:
        st.warning("Tidak ada data tren topik reguler untuk pilihan ini.")
        return

    tahun_tersedia = sorted(pd.to_datetime(subset["tahun_bulan"], format="%Y-%m", errors="coerce").dt.year.dropna().unique())
    if len(tahun_tersedia) >= 2:
        rentang_tahun = st.slider("Rentang Tahun", int(min(tahun_tersedia)), int(max(tahun_tersedia)),
                                   (int(min(tahun_tersedia)), int(max(tahun_tersedia))))
    else:
        rentang_tahun = (int(tahun_tersedia[0]), int(tahun_tersedia[0])) if tahun_tersedia else (0, 0)

    daftar_topic = sorted(subset["topic_label_auto"].dropna().unique().tolist())
    topic_dipilih = st.multiselect("Topic", daftar_topic, default=daftar_topic)

    metrik = st.radio("Tampilkan sebagai", ["Proporsi Topic", "Jumlah Dokumen"], horizontal=True)
    kolom_y = "proporsi" if metrik == "Proporsi Topic" else "jumlah_dokumen"

    tahun_dari_periode = pd.to_datetime(subset["tahun_bulan"], format="%Y-%m", errors="coerce").dt.year
    subset_filtered = subset[
        tahun_dari_periode.between(rentang_tahun[0], rentang_tahun[1])
        & subset["topic_label_auto"].isin(topic_dipilih)
    ].sort_values("tahun_bulan")

    if subset_filtered.empty:
        st.warning("Tidak ada data pada kombinasi filter ini.")
        return

    fig = px.line(
        subset_filtered, x="tahun_bulan", y=kolom_y, color="topic_label_auto", markers=True,
        labels={"tahun_bulan": "Periode (Tahun-Bulan)", kolom_y: metrik, "topic_label_auto": "Topic"},
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Lihat data tren dalam tabel"):
        st.dataframe(subset_filtered, width="stretch", hide_index=True)


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

        if "topic_label_auto" in df_f.columns:
            topic_opsi = sorted(df_f["topic_label_auto"].dropna().unique().tolist())
            topic_dipilih = st.multiselect("Topic", topic_opsi, default=topic_opsi)
            df_f = df_f[df_f["topic_label_auto"].isin(topic_dipilih)]

        if "kecamatan" in df_f.columns and df_f["kecamatan"].notna().any():
            kec_opsi = sorted(df_f["kecamatan"].dropna().unique().tolist())
            kec_dipilih = st.multiselect("Kecamatan (berita)", kec_opsi, default=kec_opsi)
            df_f = df_f[df_f["kecamatan"].isin(kec_dipilih) | df_f["kecamatan"].isna()]

        if "publisher" in df_f.columns and df_f["publisher"].notna().any():
            pub_opsi = sorted(df_f["publisher"].dropna().unique().tolist())
            pub_dipilih = st.multiselect("Publisher (berita)", pub_opsi, default=pub_opsi)
            df_f = df_f[df_f["publisher"].isin(pub_dipilih) | df_f["publisher"].isna()]

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

    column_config = {}
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
