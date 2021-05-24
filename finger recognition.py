import numpy as np
import cv2
video_capture = cv2.VideoCapture(0)
while True:
	ret, frame = video_capture.read()
	
	hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converting to hsv and taking only saturation part as it results in better accuracy 
	h,s,v = cv2.split(hsv_image)

	ret,thresh = cv2.threshold(s,50,255,cv2.THRESH_BINARY) #applying threshold to convert to binary image
	thresh = cv2.blur(thresh, (2,2)) #applying blur to smoothen the threshold lines 
	kernel = np.ones((3,3),np.uint8)
	#cv2_imshow(thresh)
	thresh = cv2.dilate(thresh,kernel,iterations = 4) #dilating in order to fill any gaps inside our hand after threshold, which are mainly due to shadows
	#cv2_imshow(thresh)
	
	contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2] 
	contours = max(contours, key=lambda x: cv2.contourArea(x)) #finding max contour points so as to avoid any noise
	for c in contours:
  		precision = 0.01 * cv2.arcLength(c, True)
  		approx = cv2.approxPolyDP(c,precision,True) #approximating contours so that the convex hull does not misunderstand any unnecessary contour point as convexity defect 
  		cv2.drawContours(frame, [approx], -1, (255,0,0), 3)
	#cv2_imshow(image)
	
	hull = cv2.convexHull(contours) #getting the convex hull points from contour points
	cv2.drawContours(frame, [hull], -1, (0,0,255), 2)
	#cv2_imshow(image)

	hull = cv2.convexHull(contours, returnPoints = False) #getting index of the hull points by setting returnPoints = False, so as to calculate convexity defects
	defects = cv2.convexityDefects(contours, hull)


	cnt = 0 
	for i in range(defects.shape[0]): #convexityDefects returns an array of 4 points [start,end, farthest point , approx distance from the farthest point] all in with respect to the convexity defect
		a, b, c, d = defects[i][0]
		start = tuple(contours[a][0])
		end = tuple(contours[b][0])
		farthest_point = tuple(contours[c][0])
		#now we calculate the sides of the triangle using distance formulae
		x = np.sqrt( (end[1] - start[1]) **2  + (end[0] - start[0]) **2 )
		y = np.sqrt( (farthest_point[1] - end[1]) **2  + (farthest_point[0] - end[0]) **2 )
		z = np.sqrt( (start[1] - farthest_point[1]) **2  + (start[0] - farthest_point[0]) **2 )
		#calculating the cosine angle by the formular angle = cos-1((a^2 + b^2 - c^2)/(2ab)) where c is the point at which cosine angle forms
		angle = np.arccos((y ** 2 + z ** 2 - x ** 2)/(2 * y * z))
		# now if the cosine angle is smaller than 90, we consider it as because angle between our fingers is naturally less than 90
		if angle < np.pi/2:
			cnt +=1 #number of convexity defects will be 1 less than the fingers raised hence incremented by 1
			cv2.circle(frame, farthest_point, 4, [0, 0, 255], -1) #drawing circle at the recognized convexity angles
	if cnt>=0: #no convexity defect will be detected if we raise only one finger, so the code will show 0 instead of 1, hence to avoid it we increament it by 1
		cnt+=1
	cv2.putText(frame, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2) #displaying the count of fingers
	cv2.imshow('Video',frame) #displaying the image
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()