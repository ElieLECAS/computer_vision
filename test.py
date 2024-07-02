import cv2
from ultralytics import YOLO

model = YOLO('yolov8_fall_detection2.pt')

source = 2

try:
    # Vérifiez si cv2.imshow() est supporté
    cv2.imshow('test', cv2.imread('test.jpg'))
    cv2.destroyAllWindows()
    imshow_supported = True
except cv2.error:
    imshow_supported = False

results = model(source, conf=0.5, save=True)

if imshow_supported:
    for result in results:
        cv2.imshow('result', result.orig_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
else:
    # Enregistrez les images si imshow n'est pas supporté
    for i, result in enumerate(results):
        cv2.imwrite(f'result_{i}.jpg', result.orig_img)

results = model(source, stream=True, conf=0.5)
for result in results:
    if imshow_supported:
        cv2.imshow('result', result.orig_img)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        # Enregistrez les images si imshow n'est pas supporté
        cv2.imwrite(f'result_{i}.jpg', result.orig_img)
