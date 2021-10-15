
import io
import picamera # Camera
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

#### THIS IS IMPORTANT FOR LIFE STREAMING ####
import logging
import socketserver
from threading import Condition
from http import server

#### THIS IS IMPORTANT FOR IMAGE PROCESSING ####
import numpy as np
import cv2

PAGE="""\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<img src="stream.mjpg" width="640" height="480" style="width:100%;height:100%;" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    frame_i = 0
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            det = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            # This is where you specify the Deep Neural Network.
            # Please put it in the same folder as the python file.
            # --> this can go at the very beginning after import cv2 in the streaming file
            interpreter = make_interpreter('face_edgetpu.tflite')
            interpreter.allocate_tensors()
            
            
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        ### The image is encoded in bytes,
                        ### needs to be converted to e.g. numpy array
                        frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8),
                                             cv2.IMREAD_COLOR)
                        
                        rects = det.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
                        
                        
                        if rects is not None:
                            #### --> needs to happen for each image ####
                            # This resizes the RGB image
                            for (x, y, w, h) in rects:
                                crop_image = frame[y:y+h, x:x+w]
                            
                                dim = (128,128)

                                resized_img = cv2.resize(crop_image, dim, interpolation = cv2.INTER_AREA)
                                resized_img = cv2.resize(crop_image, common.input_size(interpreter))
                                # Send resized image to Coral
                                common.set_input(interpreter, resized_img)

                                # Do the job
                                interpreter.invoke()
                                # Get the pose
                                pose = common.output_tensor(interpreter, 0)
                                print(interpreter.invoke())
                                if interpreter.invoke() is not None:
                                
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 20)
                            
                            
                            
                        for (x, y, w, h) in rects:
                          cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 20)
                        
                        
                        ###############
                        ## HERE CAN GO ALL IMAGE PROCESSING
                        ###############
                        
                        
                        
                        ### and now we convert it back to JPEG to stream it
                        _, frame = cv2.imencode('.JPEG', frame) 
                        
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Open the camera and stream a low-res image (width 640, height 480 px)
with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
    camera.vflip = True # Flips image vertically, depends on your camera mounting
    camera.awb_gains = (1.2, 1.5)
    camera.awb_mode = 'off'
    output = StreamingOutput() 
    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000) # port 8000
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()
