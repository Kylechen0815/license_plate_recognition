from tkinter import filedialog

import cv2
import numpy as np
import pytesseract
import os

# 設定 tesseract 執行路徑
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # ← 依你的環境改寫

# 排序四個頂點：左上、右上、右下、左下
def sort_corners(pts):
    pts = sorted(pts, key=lambda x: x[0])
    left = sorted(pts[:2], key=lambda x: x[1])
    right = sorted(pts[2:], key=lambda x: x[1])
    return np.array([left[0], right[0], right[1], left[1]], dtype="float32")

# OCR 檢查車牌格式
def is_valid_plate(text):
    return len(text) == 7 and text[:3].isalpha() and text[3:].isdigit()

# 處理單部影片並回傳合法車牌列表
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    plates = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用 HSV 的 V 通道做 CLAHE 對比強化（對白色車身效果較佳）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)

        hsv_clahe = cv2.merge((h, s, v_clahe))
        frame_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        # 灰階處理
        gray = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用較低閾值的 Canny，讓弱邊緣能被偵測出來
        edges = cv2.Canny(gray, 50, 150)

        # 形態學處理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=0)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if w > 80 and h > 10:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    sorted_pts = sort_corners(pts)
                    dst_pts = np.array([[0, 0], [199, 0], [199, 89], [0, 89]], dtype="float32")
                    M = cv2.getPerspectiveTransform(sorted_pts, dst_pts)
                    warped = cv2.warpPerspective(frame, M, (200, 80))

                    gray_plate = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    _, bin_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    bin_plate = cv2.erode(bin_plate, kernel2, iterations=0)
                    bin_plate = cv2.dilate(bin_plate, kernel2, iterations=0)
                    bin_plate = bin_plate[13:, :]
                    cv2.imshow("bin_plate",bin_plate)
                    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(bin_plate, lang='eng', config=config).strip()

                    if is_valid_plate(text):
                        plates.add(text)
                        #  顯示綠色邊框與文字
                        cv2.drawContours(edges, [approx], 0, (255, 255, 255), 5)
                        cv2.putText(edges, text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 4)

        #  顯示畫面
        #cv2.imshow("原始影像 + OCR", frame)
        cv2.imshow("Canny + Morphology", edges)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 鍵離開
            break

    cap.release()
    cv2.destroyAllWindows()
    return list(plates)



# 主程式：處理資料夾中所有影片
def main():
    folder_path = filedialog.askdirectory()
    if not os.path.isdir(folder_path):
        print("資料夾不存在")
        return

    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    if not video_files:
        print("資料夾內無影片")
        return

    for idx, filename in enumerate(video_files, start=1):
        full_path = os.path.join(folder_path, filename)
        plates = process_video(full_path)
        plate_line = ' '.join(plates)
        print(f"{idx:03d} {plate_line}")


if __name__ == "__main__":
    main()
