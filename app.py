import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 1. LOAD MODEL
# Gunakan model terbaik: MobileNetV2 AdamW
MODEL_PATH = 'model/model_mobilenetv2_adamw.keras' 
model = load_model(MODEL_PATH)

# Urutan kelas sesuai folder dataset
classes = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# 2. Fungsi Grad-CAM & Helper Functions untuk Preprocessing dan Grad-CAM
def my_preprocessing(img):
    """ Preprocessing CLAHE """
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8') if img.max() <= 1.0 else img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img.astype('float32') / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        final_layer = model.layers[-1]

        # Logika Logits vs Softmax pada layer terakhir untuk Grad-CAM model
        if hasattr(final_layer, 'activation') and final_layer.activation == tf.keras.activations.softmax:
            gradcam_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.layers[-1].input])
        else:
            gradcam_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = gradcam_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except:
        return np.zeros((7, 7))

def get_superimposed_img(img_raw, heatmap, alpha=0.6):
    """ Pewarnaan menggunakan cv2.addWeighted dan colormap jet dari matplotlib """
    img_array = np.array(img_raw)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Gunakan colormap jet dari matplotlib
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    colored_heatmap = jet_colors[heatmap_uint8] * 255
    colored_heatmap = np.uint8(colored_heatmap)
    
    # Gabungkan gambar asli dengan heatmap
    return cv2.addWeighted(img_array, 1 - alpha, colored_heatmap, alpha, 0)

# ==========================================
# 3. Informasi Penyakit (Solusi & Pencegahan)
# ==========================================
disease_info = {
    'Bacterialblight': {
        'nama_umum': 'Bacterial Blight / Hawar Daun Bakteri (Kresek)',
        'Pencegahan': (
            '1. Gunakan varietas tahan seperti Inpari 32 atau Hipa 18.\n'
            '2. Hindari pemupukan Nitrogen (Urea) yang berlebihan, terutama di musim hujan.\n'
            '3. Gunakan jarak tanam yang cukup (sistem Jajar Legowo) untuk memperbaiki sirkulasi udara.\n'
            '4. Pastikan alat pertanian dalam kondisi bersih agar tidak membawa bakteri dari lahan lain.'
        ),
        'Pengobatan': (
            '1. Jika serangan mencapai ambang batas, gunakan bakterisida berbahan aktif Tembaga Hidroksida atau Streptomisin Sulfat.\n'
            '2. Kurangi tinggi genangan air di sawah untuk menekan kelembaban.\n'
            '3. Cabut tanaman yang menunjukkan gejala "Kresek" agar bakteri tidak menyebar melalui air irigasi.'
        )
    },
    'Blast': {
        'nama_umum': 'Penyakit Blast (Pyricularia oryzae)',
        'Pencegahan': (
            '1. Hindari penggunaan pupuk Nitrogen dosis tinggi di fase awal pertumbuhan.\n'
            '2. Gunakan perlakuan benih (Seed Treatment) dengan fungisida sebelum tanam.\n'
            '3. Atur waktu tanam agar tidak bersamaan dengan puncak musim penghujan.\n'
            '4. Bakar sisa-sisa tanaman yang terinfeksi setelah panen untuk memutus siklus spora.'
        ),
        'Pengobatan': (
            '1. Aplikasi fungisida sistemik berbahan aktif Trisiklazol, Tebukonazol, atau Azoksistrobin saat muncul gejala awal di daun.\n'
            '2. Jika menyerang leher malai (potong leher), penyemprotan harus dilakukan saat padi mulai berbunga (keluar malai 5%).\n'
            '3. Gunakan agen hayati seperti jamur Trichoderma sp. untuk membantu menekan populasi patogen di tanah.'
        )
    },
    'Brownspot': {
        'nama_umum': 'Bercak Cokelat (Helminthosporium oryzae)',
        'Pencegahan': (
            '1. Pastikan tanah memiliki kandungan Kalium (K) yang cukup melalui pemupukan KCl atau NPK.\n'
            '2. Gunakan kapur pertanian (Dolomit) jika tanah terlalu asam.\n'
            '3. Perbaiki sistem drainase agar tanah tidak terendam air dalam waktu yang terlalu lama (tanah jenuh).\n'
            '4. Pilih benih dari tanaman yang sehat dan tidak membawa penyakit.'
        ),
        'Pengobatan': (
            '1. Lakukan pemupukan susulan dengan pupuk Kcl untuk memperkuat dinding sel daun.\n'
            '2. Aplikasi fungisida berbahan aktif Mankozeb atau Propineb untuk melindungi permukaan daun.\n'
            '3. Pastikan tanaman tidak mengalami cekaman kekeringan yang ekstrem.'
        )
    },
    'Tungro': {
        'nama_umum': 'Penyakit Tungro (Virus Tungro Padi)',
        'Pencegahan': (
            '1. Tanam serempak dalam satu hamparan untuk memutus siklus hidup Wereng Hijau.\n'
            '2. Gunakan varietas tahan terhadap wereng hijau seperti Inpari 36, 37, atau 46.\n'
            '3. Lakukan pengamatan rutin terhadap populasi wereng hijau sejak persemaian.\n'
            '4. Lakukan "Eradikasi" (pemusnahan) pada tanaman yang mulai menguning dan kerdil.'
        ),
        'Pengobatan': (
            '1. Fokus utama adalah mengendalikan vektor (pembawa virus) yaitu Wereng Hijau.\n'
            '2. Gunakan insektisida sistemik berbahan aktif Imidakloprid, Pimetrozin, atau Buprofezin.\n'
            '3. Perkuat daya tahan tanaman dengan pupuk daun yang mengandung unsur mikro.\n'
            '4. Penting: Tanaman yang sudah terkena virus tidak bisa disembuhkan, hanya bisa dicegah agar tidak meluas.'
        )
    }
}

# 4. Route Flask
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Simpan file yang diupload ke folder static/uploads
            filename = file.filename
            upload_path = os.path.join('static/uploads', filename)
            file.save(upload_path)

            # Preprocessing Input Gambar untuk Prediksi Model
            img_raw = tf.keras.utils.load_img(upload_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img_raw)
            img_preprocessed = my_preprocessing(img_array)
            img_input = np.expand_dims(img_preprocessed, axis=0)

            # Prediksi dengan Model
            preds = model.predict(img_input, verbose=0)[0]
            idx = np.argmax(preds)
            conf = preds[idx]
            label = classes[idx]

            # Ambil informasi solusi berdasarkan label hasil prediksi
            info = disease_info.get(label, {'Pencegahan': '-', 'Pengobatan': '-'})

            # Grad-CAM (Target layer: out_relu untuk MobileNetV2)
            heatmap = make_gradcam_heatmap(img_input, model, 'out_relu')
            
            # Superimpose (Gunakan img_array uint8 agar warna benar)
            visual_result = get_superimposed_img(img_array.astype('uint8'), heatmap)
            
            # Simpan hasil Grad-CAM ke folder static/results
            result_filename = f"gradcam_{filename}"
            result_path = os.path.join('static/results', result_filename)
            # Simpan dalam BGR karena OpenCV menggunakan BGR
            cv2.imwrite(result_path, cv2.cvtColor(visual_result, cv2.COLOR_RGB2BGR))

            return render_template('index.html', 
                                   prediction=label, 
                                   accuracy=f"{conf:.2%}", 
                                   image_url=upload_path,
                                   gradcam_url=result_path,
                                   info=info # Kirim info ke HTML
                                   )
                                   
    return render_template('index.html')
                                   
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)