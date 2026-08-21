import cv2

img = cv2.imread('cars.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car = cv2.CascadeClassifier('cars.xml')

gray = cv2.equalizeHist(gray)
cars = car.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(30, 30)
)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()