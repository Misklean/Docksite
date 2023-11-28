# app.py
import time
import io
import threading
import PIL
import numpy as np
import sys

from PIL import Image
from pyboy import PyBoy, WindowEvent
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
   emulator.send_input(WindowEvent.PRESS_ARROW_LEFT)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_ARROW_LEFT)
   return jsonify('Success - Left Arrow'), 201

@app.post("/pygame/action/rightarrow")
@cross_origin()
def add_action_rightarrow():
   emulator.send_input(WindowEvent.PRESS_ARROW_RIGHT)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
   return jsonify('Success - Right Arrow'), 201

@app.post("/pygame/action/uparrow")
@cross_origin()
def add_action_uparrow():
   emulator.send_input(WindowEvent.PRESS_ARROW_UP)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_ARROW_UP)
   return jsonify('Success - Up Arrow'), 201

@app.post("/pygame/action/downarrow")
@cross_origin()
def add_action_downarrow():
   emulator.send_input(WindowEvent.PRESS_ARROW_DOWN)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_ARROW_DOWN)
   return jsonify('Success - Down Arrow'), 201

@app.post("/pygame/action/buttona")
@cross_origin()
def add_action_buttona():
   emulator.send_input(WindowEvent.PRESS_BUTTON_A)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_BUTTON_A)
   return jsonify('Success - Button A'), 201

@app.post("/pygame/action/buttonb")
@cross_origin()
def add_action_buttonb():
   emulator.send_input(WindowEvent.PRESS_BUTTON_B)
   emulator.tick()
   emulator.send_input(WindowEvent.RELEASE_BUTTON_B)
   return jsonify('Success - Button B'), 201
