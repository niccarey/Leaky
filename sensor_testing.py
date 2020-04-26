from sht_sensor import Sht as ST

import RPi.GPIO as gp

gp.setwarnings(False)
gp.setmode(gp.BCM)

gp.setup(11, gp.OUT)
gp.setup(17, gp.IN)
gp.setup(23, gp.IN)
gp.setup(27, gp.IN)
# usage: ST(clock pin, data pin) (default voltage)

# have to use correct voltage option - default is 3.5V, otherwise
# sens = ST(clock, data, voltage=ShtVDDLevel.vdd_5v) OR try
# sens = ST(clock, data, '5V')

sens1 = ST(11, 17)
sens2 = ST(11, 27)
sens3 = ST(11, 22)

sensor_list = [sens1, sens2 , sens3]
xcount = 1

while xcount < 5:

    
    for sensi in sensor_list:
        
        try:
            print("Sensor temp: ", sensi.read_t())
            print("Sensor humidity: ", sensi.read_rh())
    
        except Exception as e:
            print(e)
    

    xcount += 1
