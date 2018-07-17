import cv2
#import numpy as np

xml = 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(xml)

for i in range(0, 2):
    nama1 = 'datasets/testing/2male/'+str(i)+'.jpg'
    image = cv2.imread(nama1, 0)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)


    for x,y,w,h in faces:
        # cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),2)
        crop = image[y:y+h, x:x+w]
    # cv2.imshow('Cropped', crop)

    resized = cv2.resize(crop, (128, 128))
    nama2 = 'datasets/testing_crop/2male/'+str(i)+'.jpg'
    cv2.imwrite(nama2, resized)
    print('data ke-', i)
    # cv2.imshow('Resized', resized)
    # cv2.imshow('Viola Jones Detect',image)

cv2.waitKey()
cv2.destroyAllWindows()

# masih error dipunyaku :(
