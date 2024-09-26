import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


webcam = cv2.VideoCapture(0)

while True:
    
    _, img = webcam.read()

   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


    cv2.imshow('Face Detection', img)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


webcam.release()
cv2.destroyAllWindows()