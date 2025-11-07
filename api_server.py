# api_server.py
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import cv2
import pydicom
import tempfile
import os

from models import load_vision_model, get_image_transform, load_segmentation_model, DEVICE
from interpret import ensure_tensor, mc_dropout_predictions
from text_pipeline import translate_en_to_ar

app = Flask(__name__)

# تحميل النماذج مرة واحدة عند بدء السيرفر
vision_model = load_vision_model(num_classes=14)
seg_model = load_segmentation_model()
transform = get_image_transform()


@app.route('/')
def home():
    return jsonify({
        "message": "🩺 MedAI Flask API جاهز — أرسل صورة أو ملف DICOM إلى /analyze"
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """تحليل صورة طبية (PNG/JPG أو DICOM)"""
    if 'image' not in request.files:
        return jsonify({"error": "الرجاء رفع الصورة تحت اسم الحقل 'image'"}), 400

    file = request.files['image']
    filename = file.filename.lower()

    # ---- تحديد نوع الملف ----
    if filename.endswith(".dcm"):
        try:
            # حفظ مؤقت للملف
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # قراءة ملف DICOM
            ds = pydicom.dcmread(tmp_path)
            img = ds.pixel_array.astype(np.float32)

            # تحويل الصورة إلى RGB (3 قنوات)
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            img_rgb = np.stack([img_norm] * 3, axis=-1)
            img_rgb = (img_rgb * 255).astype(np.uint8)

            os.remove(tmp_path)
            img_np = img_rgb
        except Exception as e:
            return jsonify({"error": f"خطأ أثناء قراءة ملف DICOM: {str(e)}"}), 500

    else:
        # ملفات الصور العادية
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_np = np.array(image)

    # ---- تجهيز الصورة للنموذج ----
    input_tensor = ensure_tensor(img_np, transform).to(DEVICE)

    # ---- تحليل بالصنف البصري ----
    vision_model.eval()
    with torch.no_grad():
        out = vision_model(input_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    # ---- أعلى 5 احتمالات ----
    top_idx = np.argsort(probs)[::-1][:5]
    findings = [{"class_id": int(i), "probability": float(probs[i])} for i in top_idx]

    # ---- تقرير ----
    en_report = "Model findings:\n" + "\n".join(
        [f"Finding_{f['class_id']}: prob={f['probability']:.3f}" for f in findings]
    )
    ar_report = translate_en_to_ar(en_report)

    # ---- استجابة JSON ----
    response = {
        "status": "success",
        "file_type": "DICOM" if filename.endswith(".dcm") else "Image",
        "findings": findings,
        "report_en": en_report,
        "report_ar": ar_report
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)