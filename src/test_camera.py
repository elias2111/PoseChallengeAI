import cv2

print("Abriendo cámara...")

# Dejamos que Windows/OpenCV elija el backend automáticamente
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

# Resolución moderada para evitar problemas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Cámara abierta")
print("Presiona Q para cerrar")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ La cámara abrió, pero no está entregando imagen")
        break

    cv2.imshow("iPhone - Pose Challenge AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()