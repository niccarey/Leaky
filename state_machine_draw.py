# printing LeakyBot state machine to file
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
import numpy as np
from LeakyBot import LeakyBot

mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)

leaky1 = LeakyBot(myMotor1, myMotor2)
    
#leaky1.show_graph()
leaky1.get_graph().draw('state_diagram.png', prog='dot')
