from pyboy import PyBoy
pyboy = PyBoy('pokemon_blue.gb')
while not pyboy.tick():
    pass
pyboy.stop()
