import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

app = Flask(__name__)

# configure  Realsense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

def gen_color_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        _, buffer = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+ buffer.tobytes() +b'\r\n' )
# Hepler to generate Depth frames

def gen_depth_frames():
    colorizer = rs.colorizer() # apply colormap for visulaization
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_color_frame = colorizer.colorize(depth_frame)
        frame = np.asanyarray(depth_color_frame.get_data())
        _, buffer = cv2.imencode('.jpg',frame)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Homepage

@app.route('/video')
def video():
    return Response(gen_color_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return  """
    <h1>Intel RealSense Streaming</h1>
    <ul>
       <li><a href="/video">RGB Stream</a></li>
       <li><a href="/depth">Depth Stream</a></li>
    <ul>
    """
@app.route('/depth')
def depth():
    return Response(gen_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


