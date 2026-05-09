import face_recognition
import cv2
import numpy as np
import datetime
from multiprocessing import Process, Value
from ctypes import c_double

def main_detect(data):
    video_capture = cv2.VideoCapture(0)

    vincent_image = face_recognition.load_image_file("emosi1.jpg")
    vincent_face_encoding = face_recognition.face_encodings(vincent_image)[0]

    wid_image = face_recognition.load_image_file("emosi2.jpg")
    wid_face_encoding = face_recognition.face_encodings(wid_image)[0]

    known_face_encodings = [
        vincent_face_encoding,
        wid_face_encoding
    ]

    known_face_name = [
        "EMOSI 1",
        "EMOSI 2"
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    people_attendance = {}

    while True:
        data.acquire()
        g3 = data.value
        data.release()

        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_name[best_match_index]

                face_names.append(name)

                # absensi
                if name not in people_attendance:
                    people_attendance[name] = datetime.datetime.now()

        process_this_frame = not process_this_frame

        # gambar kotak
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if name in people_attendance:
                attendance_time = people_attendance[name].strftime('%H:%M:%S')
                cv2.putText(frame, "Time: " + attendance_time, (left, top - 10), font, 0.6, (0, 0, 0), 1)

        cv2.putText(frame, str(float(g3)), (1070, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    y = Value(c_double, 0)
    p = Process(target=main_detect, args=(y,))
    p.start()