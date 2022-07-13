import cv2

source = [	"rtmp://ns8.indexforce.com/home/mystream",
			'rtmp://58.200.131.2:1935/livetv/cctv1'
		 ]

channel  = 2

class CaptureVideo(object):
	def net_video(self):
		# 获取网络视频流
		cam = cv2.VideoCapture(source[channel-1])
		while cam.isOpened():
			success, frame = cam.read()
			if success:
				cv2.imshow("Network", frame)

			if cv2.waitKey(1) == ord('q'):
				break

			
if __name__ == "__main__":
	capture_video = CaptureVideo()
	capture_video.net_video()