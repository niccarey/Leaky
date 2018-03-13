from sht_sensor import Sht as ST

import RPi.GPIO as gp
import time

gp.setwarnings(False)
gp.setmode(gp.BCM)

gp.setup(18, gp.OUT)

try: 
    while True:
        gp.output(18, 1) # set high
        print("off")
        time.sleep(2)
        start = time.time()
        print("on")
        while time.time() - start < 4:
            gp.output(18, 0) # set low
            time.sleep(0.05)
            gp.output(18,1)
            time.sleep(0.001)

except KeyboardInterrupt:
    gp.cleanup()
        
        
