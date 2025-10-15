import cv2
import numpy as np
import matplotlib.pyplot as plt


#img
'''
img = cv2.imread('ferrari.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show()
cv2.imwrite('ferrarigray.jpg', img)
'''

#video loading
'''
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	out.write(frame)
	cv2.imshow('frame', frame)
	cv2.imshow('gray', gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyAllWindows()
'''

#drawing
'''
img = cv2.imread('ferrari.jpg', cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (150,150), (255,255,255), 15)
#cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)
cv2.circle(img, (100,63), 55, (0,0,255), 10)

pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.polylines(img,[pts], True, (0,255,255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img, 'OpenCV Rumble', (0,130), font, 3, (200,255,255), 3, cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#image operations
'''
img = cv2.imread('ferrari.jpg', cv2.IMREAD_COLOR)
img[55,55] = [255,255,255]
px = img[55,55]

#img[100:150, 100:150] = [255,255,255]   
ferrari_wheel = img[300:500, 400:500]
img[0:200, 0:100] = ferrari_wheel 

cv2.imshow('image', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#image arithmetics and logic
'''
img1 = cv2.imread('3D-Matplotlib.png')
#img2 = cv2.imread('mainsvmimage.png')
img2 = cv2.imread('mainlogo.png')

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols] = dst 

#add = img1 + img2
#add = cv2.add(img1, img2)
#weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.imshow('dst', dst)
cv2.imshow('res', img1)
#cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#thresholding
'''
img = cv2.imread('bookpage.jpg')
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
retval3, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow('original', img)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)
cv2.imshow('gaus', gaus)
cv2.imshow('otsu', otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#Color Filtering
'''
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#hsv hue sat value
	lower_red = np.array([150,150,50])
	upper_red = np.array([180,255,150])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()
'''

#Blurring and Smoothing
'''
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#hsv hue sat value
	lower_red = np.array([150,150,50])
	upper_red = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	kernel = np.ones((15,15), np.float32)/225
	smoothed = cv2.filter2D(res, -1, kernel)
	blur = cv2.GaussianBlur(res, (15,15), 0)
	median = cv2.medianBlur(res, 15)
	bilateral = cv2.bilateralFilter(res, 15, 75, 75)

	cv2.imshow('frame', frame)
	#cv2.imshow('mask', mask)
	#cv2.imshow('res', res)
	#cv2.imshow('smoothed', smoothed)
	#cv2.imshow('blur', blur)
	cv2.imshow('median', median)
	cv2.imshow('bilateral', bilateral)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release() 
'''

#Morphological Transformation
'''
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#hsv hue sat value
	lower_red = np.array([150,150,50])
	upper_red = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	kernel = np.ones((5,5), np.uint8)
	erosion = cv2.erode(mask, kernel, iterations = 1)
	dilation = cv2.dilate(mask, kernel, iterations = 1)

	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	cv2.imshow('frame', frame)
	cv2.imshow('res', res)
	cv2.imshow('erosion', erosion)
	cv2.imshow('dilation', dilation)
	cv2.imshow('opening', opening)
	cv2.imshow('closing', closing)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release() 
'''


#Edge Detection and Gradients 
'''
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()

	laplacian = cv2.Laplacian(frame, cv2.CV_64F)
	sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize =5)
	sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize =5)
	edges = cv2.Canny(frame, 100, 200)


	cv2.imshow('original', frame)
	cv2.imshow('laplacian', laplacian)
	cv2.imshow('sobelx', sobelx)
	cv2.imshow('sobely', sobely)
	cv2.imshow('edges', edges)


	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release() 
'''

#Template Matching
'''
img_bgr = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg', 0)
w , h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
	cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

cv2.imshow('detected', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#GrabCut Foreground Extraction
'''
img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = (161,79,150,150)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()
'''

#Corner Detection
'''
img = cv2.imread('opencv-corner-detection-sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = corners.astype(int)
#corners = np.int0(corners)

for corner in corners:
	x, y = corner.ravel()
	cv2.circle(img, (x,y), 3, 255, -1)

cv2.imshow('Corner', img)
cv2.waitKey(0)
cv2.destroyAllWindowns
'''

#Feature Matching
'''
img1 = cv2.imread('opencv-feature-matching-template.jpg', 0)
img2 = cv2.imread('opencv-feature-matching-image.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2,kp2, matches[:10], None, flags = 2)
plt.imshow(img3)
plt.show()
'''

#MOG Background Reduction
'''
cap = cv2.VideoCapture('people-walking.mp4')
#cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)

	cv2.imshow('original', frame)
	cv2.imshow('fg',fgmask)

	k = cv2.waitKey(30) & 0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
'''

#Haar Cascade Objetc Detection Face & Eye
'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
	ret,img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)

	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xff 
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
'''









