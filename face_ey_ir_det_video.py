import cv2 as cv
import sys

faceCascade = cv.CascadeClassifier(r"haar_face1.xml")
eye_cascade = cv.CascadeClassifier(r'haar_eye.xml')
iris_cascade = cv.CascadeClassifier(r'haar_eye_tree_glasses.xml')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


video_capture = cv.VideoCapture(0)

while True:
 
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    
    )
    print(f'Number of faces found={len(faces)}')

 
    for (x, y, w, h) in faces:
        faces=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(faces)
        print(f' with a confidence of {confidence}')

        cv.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0),2)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color=frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        print(f'Number of eyes found={len(eyes)}')

        for (ex,ey,ew,eh) in eyes:
            eye=gray[y:y+h,x:x+w]
            label,confidence=face_recognizer.predict(eye)
            #print(f' with a confidence of {confidence}')
        
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            iris_eye_gray=eye_gray[ey:ey+eh,ex:ex+ew]
            iris_eye_color=eye_color[ey:ey+eh,ex:ex+ew]
            iris=iris_cascade.detectMultiScale(iris_eye_gray)
            print(f'Number of iris found={len(iris)}')

            for (ix,iy,iw,ih) in iris:
                iris1=gray[y:y+h,x:x+w]
                label,confidence=face_recognizer.predict(iris1)
                #print(f' with a confidence of {confidence}')
        
                cv.rectangle(iris_eye_color, (ix,iy),(ix+iw,iy+ih),(0,0,255),2)



   
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv.destroyAllWindows()