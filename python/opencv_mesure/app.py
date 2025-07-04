# install dependencies command
# $ pip install --upgrade imutils opencv-python scipy numpy

# execution command
# $ python app.py --image images/data1.png --width 0.955

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os # 파일 경로 조작을 위해 os 모듈 추가

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
# 원본 이미지의 복사본을 만들어 모든 객체가 표시될 최종 이미지로 사용
final_output_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# 저장될 이미지들을 위한 출력 디렉토리 생성
output_dir = "output_objects"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"'{output_dir}' 디렉토리가 생성되었습니다.")

object_count = 0 # 객체 카운터 초기화

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	object_count += 1 # 객체 카운트 증가

	# 현재 객체만 표시할 이미지 복사본
	current_object_image = image.copy()

	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(current_object_image, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(final_output_image, [box.astype("int")], -1, (0, 255, 0), 2) # 최종 이미지에도 그리기

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(current_object_image, (int(x), int(y)), 5, (0, 0, 255), -1)
		cv2.circle(final_output_image, (int(x), int(y)), 5, (0, 0, 255), -1) # 최종 이미지에도 그리기

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-right and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(current_object_image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(current_object_image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(current_object_image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(current_object_image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	cv2.circle(final_output_image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(final_output_image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(final_output_image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(final_output_image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(current_object_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(current_object_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	cv2.line(final_output_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(final_output_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	# draw the object sizes on the image
	cv2.putText(current_object_image, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(current_object_image, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	cv2.putText(final_output_image, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(final_output_image, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

    # 현재 객체 이미지를 고유한 이름으로 저장
	output_filename = os.path.join(output_dir, f"object_{object_count:02d}.png")
	cv2.imwrite(output_filename, current_object_image)
	print(f"객체 {object_count} 이미지가 '{output_filename}'으로 저장되었습니다.")

# 모든 객체가 표시된 최종 이미지 저장
final_output_path = "all_objects_measured.png"
cv2.imwrite(final_output_path, final_output_image)
print(f"모든 객체가 표시된 최종 이미지가 '{final_output_path}'으로 저장되었습니다.")

# 모든 창 닫기
cv2.destroyAllWindows()
print("모든 작업이 완료되었습니다. 프로그램이 종료됩니다.")
