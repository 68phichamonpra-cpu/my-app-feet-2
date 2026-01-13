from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
import math
import uuid
from datetime import datetime # Import datetime for current year in footer
from numba import njit, prange
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded_images'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
@app.route('/')


@app.route('/upload', methods=['POST'])
def upload_file():

    # -----------------------------
    # 1) ตรวจสอบไฟล์
    # -----------------------------
    if 'file' not in request.files:
        return "❌ ไม่พบไฟล์ใน request"

    file = request.files['file']

    if file.filename == '':
        return "❌ ยังไม่ได้เลือกไฟล์"

    # -----------------------------
    # 2) บันทึกไฟล์อัปโหลด
    # -----------------------------
    unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # -----------------------------
    # 3) อ่านภาพด้วย OpenCV
    # -----------------------------
    img = cv2.imread(filepath)
    if img is None:
        return "❌ ไม่สามารถอ่านไฟล์ภาพได้"

    original_img = img.copy()

    # -----------------------------
    # 4) แยกรอยเท้า (Threshold)
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # -----------------------------
    # 5) หาเท้า
    # -----------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        os.remove(filepath)
        return render_template(
            "index.html",
            error_message="❌ ไม่พบรอยเท้าในภาพ",
            now=datetime.now()
        )

    foot_contour = max(contours, key=cv2.contourArea)

    foot_mask = np.zeros_like(mask)
    cv2.drawContours(foot_mask, [foot_contour], -1, 255, thickness=-1)
    #foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    #foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel, iterations=1)




 



    # -----------------------------
    # 6) หา top / bottom ของเท้า
    # -----------------------------
    
    ys = foot_contour[:, 0, 1]
    top_y = int(np.min(ys))
    bottom_y = int(np.max(ys))
    foot_length = bottom_y - top_y
    @njit
    def count_black_pixels(arr):
        count = 0
        for i in prange(arr.shape[0]):
            for j in prange(arr.shape[1]):
                if arr[i, j] == 255:
                    count += 1
        return count
  

    if foot_length <= 0:
        os.remove(filepath)
        return "❌ ความยาวเท้าไม่ถูกต้อง"


  

       
             
                
  
    



           
              
                

    # -----------------------------
    # 7) แบ่งเป็น 3 ส่วน A B C
    # -----------------------------
    h1 = int(top_y + foot_length / 3)
    h2 = int(top_y + 2 * foot_length / 3)

    mask_A = np.zeros_like(foot_mask)
    mask_B = np.zeros_like(foot_mask)
    mask_C = np.zeros_like(foot_mask)

    mask_A[top_y:h1, :] = foot_mask[top_y:h1, :]
    mask_B[h1:h2, :] = foot_mask[h1:h2, :]
    mask_C[h2:bottom_y+1, :] = foot_mask[h2:bottom_y+1, :]

    # -----------------------------
    # 8) คำนวณพื้นที่
    # -----------------------------
    area_A = int(np.sum(mask_A > 0))
    area_B = int(np.sum(mask_B > 0))
    area_C = int(np.sum(mask_C > 0))

    total_area = area_A + area_B + area_C

    if total_area == 0:
        os.remove(filepath)
        return "❌ ไม่สามารถคำนวณพื้นที่ได้"

    arch_index = area_B / total_area

    # -----------------------------
    # 9) ตัดสินผล เท้าแบน / ปกติ
    # -----------------------------
    if arch_index > 0.26:
        foot_type = "Flat Foot (เท้าแบน)"
        foot_result = "❌ เท้าแบน"
    else:
        foot_type = "Normal Foot (เท้าปกติ)"
        foot_result = "✅ เท้าปกติ"
    if count > 0:
        areaz=area_B / count

    # -----------------------------
    # 10) วาดเส้น A B C บนภาพ
    # -----------------------------
    draw_img = original_img.copy()

    cv2.line(draw_img, (0, top_y), (draw_img.shape[1], top_y), (255, 0, 0), 2)
    cv2.line(draw_img, (0, h1), (draw_img.shape[1], h1), (0, 255, 0), 2)
    cv2.line(draw_img, (0, h2), (draw_img.shape[1], h2), (0, 255, 255), 2)
    cv2.line(draw_img, (0, bottom_y), (draw_img.shape[1], bottom_y), (255, 0, 0), 2)

    cv2.putText(draw_img, "A", (20, (top_y+h1)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(draw_img, "B", (20, (h1+h2)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(draw_img, "C", (20, (h2+bottom_y)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # -----------------------------
    # 11) บันทึกภาพผลลัพธ์
    # -----------------------------
    original_name = "original_" + unique_filename
    mask_name = "mask_" + unique_filename
    result_name = "result_" + unique_filename

    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], original_name), original_img)
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], mask_name), foot_mask)
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], result_name), draw_img)

    # -----------------------------
   # 12) เตรียมผลลัพธ์ส่งหน้าเว็บ
    # -----------------------------
    results = {
        "Count Black Pixels": f"{count_black_pixels(mask)}",
        "Areaz": f"{areaz}",



        "Area A": f"{area_A}",
        "Area B": f"{area_B}",
        "Area C": f"{area_C}",
        "Arch Index": f"{arch_index:.3f}",
        "Foot Type": foot_type,
        "Result": foot_result,
        "original_image": original_name,
        "mask_image": mask_name,
        "result_image": result_name
    }

    # -----------------------------
    # 13) ลบไฟล์อัปโหลดต้นฉบับ
    # -----------------------------
    os.remove(filepath)

    # -----------------------------
    # 14) ส่งไปแสดงผล
    # -----------------------------
    return render_template(
        "index.html",
        results=results,
        now=datetime.now()
    )
