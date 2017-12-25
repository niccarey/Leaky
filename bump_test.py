from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyACM0')
time.sleep(1)
it = util.Iterator(board)
it.start()

a5 = board.get_pin('a:5:i')

print(a5.read())

try:
    while True:
        v = 1.0 - a5.read()
        print(v)
        time.sleep(0.5)


except:
    print("Problem")
    board.exit()
