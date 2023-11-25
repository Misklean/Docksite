# app.py
import time
import io
import threading
import PIL

from PIL import Image
from pyboy import PyBoy
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def background_task():
    while not emulator.tick():
        pass
    emulator.stop()

emulator = PyBoy('games/pokemon_blue.gb')
background_thread = threading.Thread(target=background_task)
background_thread.start()

@app.get('/pygame')
@cross_origin()
def getGameScreenData():
    screen_image = emulator.screen_image()
    screen_image = screen_image.resize((800, 720))
    img_byte_arr = io.BytesIO()
    screen_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(img_byte_arr, mimetype='application/octet-stream')

@app.post("/pygame/action/leftarrow")
@cross_origin()
def add_action_leftarrow():
    return jsonify('Success - Left Arrow'), 201

@app.post("/pygame/action/rightarrow")
@cross_origin()
def add_action_rightarrow():
    return jsonify('Success - Right Arrow'), 201

@app.post("/pygame/action/uparrow")
@cross_origin()
def add_action_uparrow():
    return jsonify('Success - Up Arrow'), 201

@app.post("/pygame/action/downarrow")
@cross_origin()
def add_action_downarrow():
    return jsonify('Success - Down Arrow'), 201
