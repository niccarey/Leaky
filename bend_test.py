from pyfirmata import Arduino, util
import time
import serial

test = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)
printout = test.read("5".encode('ascii'))

print(printout)
