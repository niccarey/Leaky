from transitions import Machine
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
import time
import random

# could extend this by bringing in motor functions (lowpass filter, etc)

class LeakyBot(object):
    # each leaky bot should have two motors, left and right, 
    # and a standard set of states and transitions
    # Conditions may be unecessary and slowing things down
    
    states = ['waiting', 'turning', 'sensing', 'driving', 'deposit', 'backup', 'go_home']
    
    transitions = [ {'trigger':'button_push', 'source':'waiting', 'dest':'turning'},
        {'trigger':'button_push', 'source':'deposit', 'dest':'backup'},
        {'trigger':'walls_balanced', 'source':'turning', 'dest':'sensing', 'prepare':'set_loop', 'conditions':['do_have_block', 'do_high_humidity']}, 
        {'trigger':'humidity_maintained', 'source': 'sensing', 'dest':'driving', 'conditions':'do_high_humidity'},
        {'trigger':'stop_driving', 'source':'driving', 'dest':'sensing'},
        {'trigger':'low_humidity', 'source':'sensing', 'dest':'turning', 'prepare':'set_turn_status'},
        {'trigger':'wall_found', 'source':'turning', 'dest':'deposit', 'prepare':'set_deposit_direction', 'conditions':'do_have_block', 'unless':'do_high_humidity'},
        {'trigger':'reached_wall', 'source':'deposit', 'dest':'backup'},
        {'trigger':'home_spotted', 'source':'backup', 'dest':'go_home', 'unless':'do_have_block'},
        {'trigger':'close_to_home', 'source':'go_home', 'dest':'waiting'},
        {'trigger':'button_push', 'source':'go_home', 'dest':'waiting'} ]
        
        
    high_humidity = True
    have_block = False
    speed = 0
    threshold_state = 'leq'
    deposit_direction = 'left'
    start_turning_frame = 0
    
    driving_clock = 0
    sensing_clock = 0
    
    # won't need this once humidity sensing is added
    sensor_loop = 0
    sensor_loop_max = 4
    
    def __init__(self, motor_left, motor_right):
        # uniform initialization functions
        self.machine = Machine(model=self, states=LeakyBot.states, transitions=LeakyBot.transitions, initial='waiting')
        self.direction = 'left' # not sure if I want this now
        
        # attach motors to object
        self.motor_left = motor_left
        self.motor_right = motor_right
        #self.get_graph().draw('state_diagram.png', prog='dot')
        
    
    def do_have_block(self):
        return(self.have_block)
        
    def do_high_humidity(self):
        return(self.high_humidity)
        
    def on_enter_turning(self):
        print(self.state)
        des_speed = self.speed
        self.set_motor_values(des_speed, des_speed)
                
        
    def on_exit_turning(self):
        # Pause briefly before transition
        self.set_motor_values(0,0)
        time.sleep(2)
        
        
    def on_enter_driving(self):
        print(self.state)
        self.direction = 'fwd' 
        self.set_motor_values(self.speed, self.speed)
        self.driving_clock = time.time()
        
    def on_exit_driving(self):
        self.set_motor_values(0,0)
        

    def on_enter_sensing(self):
        print(self.state)
        # control the motors here.
        self.set_motor_values(0,0)
        self.sensing_clock = time.time()
        self.sensor_loop += 1
                
    
    def set_loop(self):
        self.sensor_loop = 0
        
        
    def set_deposit_direction(self):
        if self.deposit_direction == 'left':
            self.threshold_state = 'leq'
                            
        else:
            self.threshold_state = 'geq'


    def set_turn_status(self):
        turn_dir = bool(random.getrandbits(1))

        if turn_dir:
            self.direction = 'left'
            self.deposit_direction = 'left'
            self.threshold_state = 'geq'

        else:
            self.direction = 'right'
            self.deposit_direction = 'right'
            self.threshold_state = 'leq'
            
        print("Turning to deposit: ", self.deposit_direction)        
        self.set_motor_values(0,0)
        time.sleep(2)
        
    
    def on_enter_waiting(self):
        print(self.state)
        self.set_motor_values(0,0)
        
    
    def on_exit_waiting(self):
        time.sleep(2)
        self.have_block = True
        self.high_humidity = True
        self.start_turning_frame = 0
        #turn_dir = bool(random.getrandbits(1))        
        #if turn_dir:
        #    self.direction = 'left'
        #
        #else:
        self.direction = 'right'
        
    def on_enter_deposit(self):
        print(self.state)
        self.direction = 'fwd' 
        des_speed = self.speed
        self.set_motor_values(des_speed, des_speed)
        
    def on_exit_deposit(self):
        self.have_block = False 

        self.set_motor_values(0,0)
        time.sleep(2)
            
        
    def on_enter_backup(self):
        print(self.state)
        self.direction = 'rev'
        # different to a normal turn, so:
        if self.deposit_direction == 'left':
            self.set_motor_values(self.speed, 0)
            
        else:
            self.set_motor_values(0, self.speed)

            
    def on_exit_backup(self):
        self.set_motor_values(0,0)
        time.sleep(2)
        
  
    def on_enter_go_home(self):
        print(self.state)
        self.direction = 'fwd'
        self.high_humidity = True
        des_speed = self.speed
        self.set_motor_values(des_speed, des_speed)

  
    def set_motor_values(self, left_speed, right_speed):
        ml = self.motor_left
        mr = self.motor_right
        check_dir = self.direction
        
        ml.setSpeed(left_speed)
        mr.setSpeed(right_speed)

        if (left_speed == 0) and (right_speed == 0):
            ml.run(Adafruit_MotorHAT.RELEASE)
            mr.run(Adafruit_MotorHAT.RELEASE)
        
        elif check_dir == 'rev':
            ml.run(Adafruit_MotorHAT.BACKWARD)
            mr.run(Adafruit_MotorHAT.BACKWARD)
        
        elif check_dir == 'left':
            ml.run(Adafruit_MotorHAT.BACKWARD)
            mr.run(Adafruit_MotorHAT.FORWARD)

        elif check_dir == 'right':
            ml.run(Adafruit_MotorHAT.FORWARD)
            mr.run(Adafruit_MotorHAT.BACKWARD)
            
        elif check_dir == 'fwd':
            ml.run(Adafruit_MotorHAT.FORWARD)
            mr.run(Adafruit_MotorHAT.FORWARD)
            
        
            
        
    def have_block(self):
        return self.have_block
        
    def high_humidity(self):
        return self.high_humidity
        
        

