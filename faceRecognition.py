import cv2
import face_recognition as fr
import numpy as np

knownNames = {}


# noinspection SpellCheckingInspection
def loadFaces():
    global knownNames
    barakObamaEncoded = fr.face_encodings(fr.load_image_file("Images/Barak Obama.jpg"))[0]
    beniGantzEncoded = fr.face_encodings(fr.load_image_file("Images/Beni Gantz.jpg"))[0]
    bibiNetanyahuEncoded = fr.face_encodings(fr.load_image_file("Images/Bibi Netanyahu.jpg"))[0]
    donaldTrumpEncoded = fr.face_encodings(fr.load_image_file("Images/Donald Trump.jpg"))[0]
    vladimirPutinEncoded = fr.face_encodings(fr.load_image_file("Images/Vladimir Putin.jpg"))[0]

    knownNames = {"Barak Obama": barakObamaEncoded, "Beni Gantz": beniGantzEncoded,
                  "Bibi Netanyahu": bibiNetanyahuEncoded, "Donald Trump": donaldTrumpEncoded,
                  "Vladimir Putin": vladimirPutinEncoded}


def faceRecognizer(path):
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    assert ret, "failed reading image"
    height, width, channels = frame1.shape

    face_locations = []
    faceNames = []
    checkFrame = True

    while cap.isOpened():
        smallFrame = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)
        rgb_smallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)
        if checkFrame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = fr.face_locations(rgb_smallFrame)
            face_encodings = fr.face_encodings(rgb_smallFrame, face_locations)

            faceNames = []
            for Encoded in face_encodings:
                matches = fr.compare_faces(knownNames.values(), Encoded)
                name = "Unknown"

                faceDistances = fr.face_distance(knownNames.values(), Encoded)
                bestMatchIndex = np.argmin(faceDistances)
                if matches[bestMatchIndex]:
                    name = knownNames.keys()[bestMatchIndex]

                faceNames.append(name)
        checkFrame = not checkFrame

        for (top, right, bottom, left), name in zip(face_locations, faceNames):
            top = int(top * 4)
            right = int(right * 4)
            bottom = int(bottom * 4)
            left = int(left * 4)

            if name == "Unknown":
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            else:
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame1, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.putText(frame1, 'Press Q to quit', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video', frame1)
        ret, frame1 = cap.read()

        if cv2.waitKey(40) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
