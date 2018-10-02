# software versions, part of testing utility

# Ensure OpenCV > 3.4.2
import cv2 as cv
OPENCV_VERSION = (3,4,2)
try:
	cv_version = tuple(map(int,cv.__version__.split('.')))
	assert(cv_version >= OPENCV_VERSION)
except AssertionError as ex:
	print('[-] Minimum is OpenCV 3.4.2. You are using {}'.format(cv.__version__))
	raise


# Ensure PyTorch == 0.3.0
PYTORCH_VERSION = (0,3,0)