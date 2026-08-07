import cv2

from preprocessing.extract_frame import extractframe

reader = extractframe(
    "data/raw/DFD_original sequences/01__exit_phone_room.mp4"
)

for frame_number, frame in reader:

    print(frame_number)

    cv2.imshow("Video", frame)

    if cv2.waitKey(10) == 27:
        break

reader.release()

cv2.destroyAllWindows()
