from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyACM0')
time.sleep(1)
it = util.Iterator(board)
it.start()

a0 = board.get_pin('a:0:i')
a1 = board.get_pin('a:1:i')
a2 = board.get_pin('a:2:i')

print('ready')

try:
    while True:
        v = a0.read()
        print(1000*v)
        v = a1.read()
        print(1000*v)
        v = a2.read()
        print(1000*v)
        time.sleep(0.5)
        


except:
    print("Problem")
    board.exit()
