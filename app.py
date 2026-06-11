import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable(name="CrossLayer")
class CrossLayer(Layer):
    def __init__(self, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = int(input_shape[0][-1])

        self.w = self.add_weight(
            shape=(dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="cross_w"
        )

        self.b = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="cross_b"
        )

        super(CrossLayer, self).build(input_shape)

    def call(self, inputs):
        x0, xl = inputs
        xlw = tf.matmul(xl, self.w)
        cross = x0 * xlw + self.b + xl
        return cross


# KONFIGURASI

st.set_page_config(
    page_title="Prediksi Kelayakan Giling Tebu",
    page_icon="🌾",
    layout="wide"
)

FEATURES = ["Jarak", "Panjang", "Drata", "Keliling", "Berat"]

MODEL_PATHS = {
    "MLP": "model_mlp90_10-clear.keras",
    "DCN": "model_dcn9010-random.keras"
}

SCALER_PATH = "scaler.pkl"

SAMPLE_FILES = {
    "Data Sampel 378 Baris": "tebu 378(20) csv.csv",
    "Data Sampel 284 Baris": "tebu 284(15) csv.csv",
    "Data Sampel 189 Baris": "tebu 189(10) csv.csv"
}



# LOAD MODEL DAN SCALER

@st.cache_resource
def load_keras_model(model_path):
    return tf.keras.models.load_model(
        filepath=model_path,
        custom_objects={"CrossLayer": CrossLayer},
        compile=False,
        safe_mode=False
    )


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)



# FUNGSI PREDIKSI

def prediksi_data(df_input, model, scaler):
    X = df_input[FEATURES].copy()

    # Pastikan semua data numerik
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        raise ValueError("Terdapat data kosong atau bukan angka pada kolom fitur.")

    X_scaled = scaler.transform(X)

    prob = model.predict(X_scaled, verbose=0).ravel()

    kelas = (prob >= 0.5).astype(int)

    status = np.where(
        kelas == 1,
        "Layak Giling",
        "Tidak Layak Giling"
    )

    confidence = np.where(
        kelas == 1,
        prob * 100,
        (1 - prob) * 100
    )

    hasil = df_input.copy()
    hasil["Probabilitas"] = prob
    hasil["Kelas"] = kelas
    hasil["Status"] = status
    hasil["Confidence (%)"] = confidence.round(2)

    return hasil


# STYLE CSS

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    margin-top: 4px;
    margin-bottom: 25px;
    color: #cfcfcf;
}

.card {
    padding: 18px;
    border-radius: 14px;
    background-color: #1e1f29;
    border: 1px solid #333542;
    margin-bottom: 15px;
}

.result-success {
    padding: 18px;
    border-radius: 14px;
    background-color: #113d2a;
    border: 1px solid #1fa36a;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}

.result-error {
    padding: 18px;
    border-radius: 14px;
    background-color: #4a1f24;
    border: 1px solid #d94c5c;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}

[data-testid="stMetric"] {
    background-color: #f7f9fc;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #d9dee8;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] div {
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)



# HEADER

st.markdown("""
<div style='text-align:center;'>

<h1 style='
margin-bottom:8px;
font-size:52px;
font-weight:700;
'>
Prediksi Kelayakan Giling Tebu
</h1>

<p style='
font-size:22px;
font-weight:500;
color:#444444;
margin-top:0;
'>
Sistem prediksi berbasis model MLP dan DCN
</p>

</div>
""", unsafe_allow_html=True)



# PANEL KONTROL

control_col1, control_col2 = st.columns([1, 1])

with control_col1:
    model_choice = st.selectbox(
        "Pilih Model Prediksi",
        ["MLP", "DCN"],
        key="model_choice"
    )

with control_col2:
    metode_input = st.radio(
        "Pilih Metode Input Data",
        ["Input Manual", "Upload File CSV / Excel"],
        horizontal=True,
        key=f"metode_input_{model_choice}"
    )

model_path = MODEL_PATHS[model_choice]

if not os.path.exists(model_path):
    st.error(f"File model tidak ditemukan: {model_path}")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"File scaler tidak ditemukan: {SCALER_PATH}")
    st.stop()

model = load_keras_model(model_path)
scaler = load_scaler()


# =========================
# INPUT MANUAL
# =========================
if metode_input == "Input Manual":

    st.markdown("### 📥 Input Data Sampel")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        jarak = st.number_input("Jarak (cm)", min_value=0.0, step=0.1, key=f"jarak_{model_choice}")

    with col2:
        panjang = st.number_input("Panjang (cm)", min_value=0.0, step=0.1, key=f"panjang_{model_choice}")

    with col3:
        drata = st.number_input("Drata", min_value=0.0, step=0.1, key=f"drata_{model_choice}")

    with col4:
        keliling = st.number_input("Keliling (cm)", min_value=0.0, step=0.1, key=f"keliling_{model_choice}")

    with col5:
        berat = st.number_input("Berat (g/cm)", min_value=0.0, step=0.1, key=f"berat_{model_choice}")

    tombol_prediksi = st.button("🔍 Prediksi Sampel", use_container_width=True)

    if tombol_prediksi:
        
        if jarak == 0 and panjang == 0 and drata == 0 and keliling == 0 and berat == 0:
            st.error("Data input masih kosong. Silakan isi nilai fitur terlebih dahulu.")
            st.stop()

        if jarak <= 0 or panjang <= 0 or drata <= 0 or keliling <= 0 or berat <= 0:
            st.error("Semua fitur harus diisi dengan nilai lebih dari 0.")
            st.stop()

        df_manual = pd.DataFrame({
            "Jarak": [jarak],
            "Panjang": [panjang],
            "Drata": [drata],
            "Keliling": [keliling],
            "Berat": [berat]
        })

        try:
            hasil = prediksi_data(df_manual, model, scaler)

            left_box, right_box = st.columns([1.2, 1])

            with left_box:
                st.markdown("### 📋 Data Sampel")
                st.dataframe(df_manual, use_container_width=True, hide_index=True)

            with right_box:
                st.markdown("### 🎯 Hasil Prediksi")

                kelas = hasil.loc[0, "Kelas"]
                confidence = hasil.loc[0, "Confidence (%)"]
                prob = hasil.loc[0, "Probabilitas"]

                if kelas == 1:
                    st.markdown(
                        '<div class="result-success">✅ Layak Giling</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-error">❌ Tidak Layak Giling</div>',
                        unsafe_allow_html=True
                    )

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("Model", model_choice)

                with m2:
                    st.metric("Kelas", int(kelas))

                with m3:
                    st.metric("Confidence", f"{confidence:.2f}%")

                st.progress(int(confidence))
                st.caption(f"Probabilitas kelas 1: {prob:.4f}")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


# =========================
# UPLOAD FILE
# =========================
else:

    upload_col, info_col = st.columns([1, 1])

    with upload_col:
        st.markdown("### 📤 Upload Data Uji")

        sumber_data = st.radio(
            "Pilih Sumber Data",
            ["Upload File Sendiri", "Gunakan Data Sampel"],
            horizontal=True,
            key=f"sumber_data_{model_choice}"
        )

        uploaded_file = None
        selected_sample = None

        if sumber_data == "Upload File Sendiri":
            uploaded_file = st.file_uploader(
                "Upload file CSV atau Excel",
                type=["csv", "xlsx"],
                key=f"upload_{model_choice}"
            )

        else:
            selected_sample = st.selectbox(
                "Pilih Data Sampel",
                list(SAMPLE_FILES.keys()),
                key=f"sample_file_{model_choice}"
            )

            sample_path = SAMPLE_FILES[selected_sample]

            if os.path.exists(sample_path):
                with open(sample_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Data Sampel",
                        data=file,
                        file_name=sample_path,
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error(f"File sampel tidak ditemukan: {sample_path}")
                st.stop()

    with info_col:
        st.markdown("### ℹ️ Format File")
        st.info(
            "File boleh memiliki kolom tambahan seperti No, ID, Nomor, atau Kode. "
            "Sistem hanya mengambil kolom: Jarak, Panjang, Drata, Keliling, dan Berat."
        )

    if uploaded_file is not None or selected_sample is not None:

        try:
            if sumber_data == "Upload File Sendiri":
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
            else:
                sample_path = SAMPLE_FILES[selected_sample]
                df_upload = pd.read_csv(sample_path)

            missing_cols = [col for col in FEATURES if col not in df_upload.columns]

            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
                st.info("Pastikan file memiliki kolom: Jarak, Panjang, Drata, Keliling, Berat")
                st.stop()

            preview_col, action_col = st.columns([1.5, 1])

            with preview_col:
                st.markdown("### 📄 Preview Data")
                st.dataframe(df_upload.head(10), use_container_width=True)

            with action_col:
                st.markdown("### ⚙️ Informasi Data")
                st.metric("Jumlah Data", len(df_upload))
                st.metric("Jumlah Kolom", len(df_upload.columns))
                st.metric("Model Aktif", model_choice)

                tombol_prediksi_file = st.button(
                    "🔍 Prediksi Semua Data",
                    use_container_width=True
                )

            if tombol_prediksi_file:

                hasil = prediksi_data(df_upload, model, scaler)

                jumlah_layak = (hasil["Kelas"] == 1).sum()
                jumlah_tidak_layak = (hasil["Kelas"] == 0).sum()

                st.markdown("### 📊 Ringkasan Prediksi")

                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric("Total Data", len(hasil))

                with col_b:
                    st.metric("Layak Giling", jumlah_layak)

                with col_c:
                    st.metric("Tidak Layak Giling", jumlah_tidak_layak)

                with col_d:
                    persen_layak = (jumlah_layak / len(hasil)) * 100
                    st.metric("Persentase Layak", f"{persen_layak:.2f}%")

                st.markdown("### 🎯 Hasil Prediksi Data Uji")
                st.dataframe(hasil, use_container_width=True)

                csv = hasil.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Hasil Prediksi CSV",
                    data=csv,
                    file_name=f"hasil_prediksi_{model_choice}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Terjadi error saat membaca atau memproses file: {e}")

