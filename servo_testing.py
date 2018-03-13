import RPi.GPIO as gp
import time

gp.setmode(gp.BCM)
gp.setup(18, gp.OUT)
p = gp.PWM(18, 50) # 50Hz for most servos

# duty cycle map: 1ms = left, 1.5ms = middle, 2ms = right (theoretically)
# dc calculations: dcmap/20 * 100

p.start(5) # hopefully left
time.sleep(0.1)

try:
    while True:
        p.ChangeDutyCycle(10)
        time.sleep(1)
        p.ChangeDutyCycle(5)
        time.sleep(1)
        
except KeyboardInterrupt:
    p.stop()
    gp.cleanup()
    
