from deepforest import main
import cv2

model = main.deepforest()
model.use_release()

img_original=cv2.imread('trees/trees6.jpg')

# Predict image
img = model.predict_image(path="./trees/trees6.jpg")

xmin_values = img['xmin'].astype(int).tolist()
ymin_values = img['ymin'].astype(int).tolist()
xmax_values = img['xmax'].astype(int).tolist()
ymax_values = img['ymax'].astype(int).tolist()
score=img['score'].astype(float).tolist()
cnt=0
cv2.imshow('original img',img_original)
for xmin, ymin, xmax, ymax,score in zip(xmin_values, ymin_values, xmax_values, ymax_values,score):
         
            cnt=cnt+1
   
            cv2.rectangle(img_original, (xmin, ymin), (xmax, ymax), (0,0,0), 1)
    
cv2.imshow('Detected Trees', img_original)
print(f'number of trees detected is {cnt}')

cv2.waitKey(0)
cv2.destroyAllWindows()


