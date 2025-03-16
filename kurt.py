import cv2
import numpy as np
import mediapipe as mp
import time
import pygame  # Ses için

# pygame mixer'ı başlat
pygame.mixer.init()
wolf_sound = pygame.mixer.Sound("wolf_howl.wav")  # Kurt uluması sesi dosyası

# Kurt resimlerini yükleyelim
wolf_image_left = cv2.imread("kurt_left.png", cv2.IMREAD_UNCHANGED)  # Sol taraf için kurt
wolf_image_right = cv2.imread("kurt_right.png", cv2.IMREAD_UNCHANGED)  # Sağ taraf için kurt

# Alfa kanalı yoksa ekleyelim
if wolf_image_left.shape[-1] == 3:
    b, g, r = cv2.split(wolf_image_left)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    wolf_image_left = cv2.merge((b, g, r, alpha))

if wolf_image_right.shape[-1] == 3:
    b, g, r = cv2.split(wolf_image_right)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    wolf_image_right = cv2.merge((b, g, r, alpha))

# MediaPipe Face Mesh ve Pose ayarı
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Kurtların başlangıç ve hedef konumları
wolf_pos_left = [50, 50]
wolf_pos_right = [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 200, 50]
target_pos_left = [50, 50]
target_pos_right = [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 200, 50]

# Kaybolma süresi (kurtlar hedefe ulaştıktan sonra kaybolacak süre)
wolf_hide_time = 5  # saniye
last_hide_time = None  # Ses kontrol bayrağı (her "O" tetiklendiğinde ses yeniden çalınsın)
sound_played = False

# Kayma etkisini ekleyelim (random kayma)
def apply_screen_slide(frame, intensity=5):
    """Ekrana kaydırma efekti ekler"""
    height, width, _ = frame.shape
    x_offset = np.random.randint(-intensity, intensity)  # Yatay kayma
    y_offset = np.random.randint(-intensity, intensity)  # Dikey kayma

    # Kaydırma matrisini uygulayalım
    M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])  # Kaydırma matrisi
    frame_shaken = cv2.warpAffine(frame, M, (width, height))
    return frame_shaken

def overlay_image(background, overlay, x, y, alpha=0.7):
    """
    Overlay image onto background with blending.
    Ensure that the overlay and background regions have the same shape.
    """
    h, w, c = overlay.shape
    if y + h > background.shape[0] or x + w > background.shape[1]:
        print("Warning: Overlay exceeds background bounds!")
        return

    # Resize overlay if necessary to match the target region
    overlay_resized = cv2.resize(overlay, (w, h))

    # Loop over each channel (assuming 3 channels for color image)
    for c in range(3):  # Assuming a 3-channel image (BGR)
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay_resized[:, :, c]

def overlay_with_glow(background, overlay, x, y, glow_factor=0.5, blur_kernel=(15,15), scale_up=1.2):
    """
    Adds a glow effect by first resizing and blurring the overlay, then applying it to the background.
    """
    h, w, _ = overlay.shape
    glow_overlay = cv2.resize(overlay, (int(w * scale_up), int(h * scale_up)))
    
    # Calculate position to center the glow effect on (x, y)
    glow_x = x - int((glow_overlay.shape[1] - w) / 2)
    glow_y = y - int((glow_overlay.shape[0] - h) / 2)

    glow_overlay = cv2.GaussianBlur(glow_overlay, blur_kernel, 0)
    
    # Resize the overlay image to fit the target region on the frame
    if glow_overlay.shape[-1] == 4:
        glow_overlay[:, :, 3] = (glow_overlay[:, :, 3].astype(np.float32) * glow_factor).astype(np.uint8)
    
    # Make sure both glow_overlay and target region have the same size
    glow_overlay_resized = cv2.resize(glow_overlay, (w, h))  # Resize glow overlay to match the original overlay size
    
    # Apply glow effect on the background (using alpha blending)
    overlay_image(background, glow_overlay_resized, glow_x, glow_y, alpha=0.5)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kameradan alınan görüntüyü aynalayalım
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    results_pose = pose.process(rgb_frame)

    # Ekran kaydırma efekti aktifse, kaydırma efekti ekleyelim
    if last_hide_time is not None:
        frame = apply_screen_slide(frame, intensity=10)  # Kayma şiddetini buradan ayarlayabilirsiniz

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_opening = abs(lower_lip.y - upper_lip.y)

            if mouth_opening > 0.04:  # "O" harfi tetiklendiğinde
                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark
                    left_shoulder = landmarks[11]   # Sol omuz
                    right_shoulder = landmarks[12]    # Sağ omuz

                    height, width, _ = frame.shape
                    left_shoulder_x = int(left_shoulder.x * width)
                    left_shoulder_y = int(left_shoulder.y * height)
                    right_shoulder_x = int(right_shoulder.x * width)
                    right_shoulder_y = int(right_shoulder.y * height)

                    # Kurtların omuzlara doğru hareketi
                    wolf_pos_left[0] += (left_shoulder_x - wolf_pos_left[0] - 75) * 0.05
                    wolf_pos_left[1] += (left_shoulder_y - wolf_pos_left[1] - 75) * 0.05
                    wolf_pos_right[0] += (right_shoulder_x - wolf_pos_right[0] - 75) * 0.05
                    wolf_pos_right[1] += (right_shoulder_y - wolf_pos_right[1] - 75) * 0.05

                    # Kurt resimlerini 150x150 boyutunda yeniden boyutlandır
                    wolf_resized_left = cv2.resize(wolf_image_left, (150, 150))
                    wolf_resized_right = cv2.resize(wolf_image_right, (150, 150))

                    # Glow efektiyle kurtları ekrana yerleştir
                    overlay_with_glow(frame, wolf_resized_left, int(wolf_pos_left[0]), int(wolf_pos_left[1]),
                                      glow_factor=0.5, blur_kernel=(15,15), scale_up=1.2)
                    overlay_with_glow(frame, wolf_resized_right, int(wolf_pos_right[0]), int(wolf_pos_right[1]),
                                      glow_factor=0.5, blur_kernel=(15,15), scale_up=1.2)

                    # Ses tetikleme
                    if not sound_played:
                        wolf_sound.play()
                        sound_played = True

                if last_hide_time is None:
                    last_hide_time = time.time()
            else:
                # Ağız kapanınca konumlar sıfırlanır ve ses bayrağı resetlenir
                wolf_pos_left = target_pos_left.copy()
                wolf_pos_right = target_pos_right.copy()
                last_hide_time = None
                sound_played = False

    # Eğer kurtlar ekranda 5 saniyeden fazla kalmışsa, efekt sıfırlansın
    if last_hide_time and (time.time() - last_hide_time) > wolf_hide_time:
        wolf_pos_left = target_pos_left.copy()
        wolf_pos_right = target_pos_right.copy()
        last_hide_time = None
        sound_played = False

    # Eğer "O" tetiklenmişse, etrafı karanlıklaştıran bir overlay uygulayalım
    if last_hide_time is not None:
        # Boş bir maske oluştur (0: koyu, 1: parlak)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        cv2.circle(mask, (int(wolf_pos_left[0] + 75), int(wolf_pos_left[1] + 75)), 100, 1, -1)
        cv2.circle(mask, (int(wolf_pos_right[0] + 75), int(wolf_pos_right[1] + 75)), 100, 1, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        multiplier = 0.3 + 0.7 * mask
        darkened_frame = (frame.astype(np.float32) * multiplier[..., np.newaxis]).astype(np.uint8)
        cv2.imshow("Wolf Effect", darkened_frame)
    else:
        cv2.imshow("Wolf Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


















