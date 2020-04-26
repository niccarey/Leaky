from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyACM0')
time.sleep(0.5)
it = util.Iterator(board)
it.start()

flex1 = board.get_pin('a:0:i')
flex2 = board.get_pin('a:2:i')

fcount = 0
running = True

# constants
VCC = 3.3
R_DIV = 10000 # think I used a 10k resistor

def flex_sensor_calib(flexpin1, flexpin2, vcc):
    flexcount = 0
    f1_sum = 0
    f2_sum = 0
    while flexcount < 10:
        time.sleep(0.3)
        f1_v = flexpin1.read()*VCC
        f2_v = flexpin2.read()*VCC
        print(f1_v, f2_v)
        f1_sum += f1_v
        f2_sum += f2_v
        flexcount += 1

    return f1_sum/10, f2_sum/10


f1_av, f2_av = flex_sensor_calib(flex1, flex2, VCC)

print('Average mean values:', f1_av, f2_av)

fcount = 0
stime = time.time()
with open("test_write.txt", "a") as storefile:
    while fcount < 20:
        time.sleep(0.3)
        f1_reading =150*( flex1.read()*VCC - f1_av)
        f2_reading = 100*( flex2.read()*VCC - f2_av)
        storefile.write(str(time.time()-stime) + " Sensors: " +str(f1_reading) +", " + str(f2_reading) + "\n")
        fcount += 1
        
        

