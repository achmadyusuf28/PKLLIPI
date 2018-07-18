from picamera import PiCamera
import subprocess

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
filename = 'videotraining'

def rekam():
	camera.start_preview()
	camera.start_recording(filename+'.h264')
	print 'recording...'
	
def stop():
	camera.stop_preview()
	camera.stop_recording()
	print 'stop recording'
	print 'converting to mp4'
	command = "MP4Box -add {} {}.mp4".format(filename+'.h264', filename)
	try:
		output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
	except subprocess.CalledProcessError as e:
		print('FAIL:\ncmd:{}\noutput:{}'.format(e.cmd, e.output))
	
def option(str):
	if str == "q":
		stop()
	else :
		rekam()

if __name__ == '__main__':
	text = raw_input('perintah : ')
	option(text)
	text = raw_input('perintah : ')
	option(text)
	
	