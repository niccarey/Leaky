from transitions import Machine
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
import numpy as np
import time
import random

# could extend this by bringing in motor functions (lowpass filter, etc)

class LeakyBot(object):
    # each leaky bot should have two motors, left and right, 
    # and a standard set of states and transitions
 
    states = ['waiting', 'turning', 'sensing', 'driving', 'deposit', 'backup', 'go_home']
    
    transitions = [ {'trigger':'button_push', 'source':'waiting', 'dest':'turning'},
        {'trigger':'button_push', 'source':'deposit', 'dest':'backup'},
        {'trigger':'button_push', 'source':'turning', 'dest':'backup', 'unless':'do_high_humidity'},
        {'trigger':'walls_balanced', 'source':'turning', 'dest':'sensing', 'prepare':'set_loop', 'conditions':['do_have_block', 'do_high_humidity']}, 
        {'trigger':'humidity_maintained', 'source': 'sensing', 'dest':'driving', 'conditions':'do_high_humidity'},
        {'trigger':'stop_driving', 'source':'driving', 'dest':'sensing'},
        {'trigger':'low_humidity', 'source':'sensing', 'dest':'turning', 'prepare':'set_turn_status'},
        {'trigger':'wall_found', 'source':'turning', 'dest':'deposit', 'conditions':'do_have_block', 'unless':'do_high_humidity'},
        {'trigger':'reached_wall', 'source':'deposit', 'dest':'backup'},
        {'trigger':'home_spotted', 'source':'backup', 'dest':'go_home', 'unless':'do_have_block'},
        {'trigger':'close_to_home', 'source':'go_home', 'dest':'waiting'},
        {'trigger':'button_push', 'source':'go_home', 'dest':'waiting'},
        {'trigger': 'static_visuals', 'source': 'driving', 'dest': 'backup'},
        {'trigger': 'static_visuals', 'source': 'turning', 'dest':'backup', 'unless': 'do_high_humidity'}]
        
        
    start_turning_frame = 0
    
    
    def __init__(self, motor_left, motor_right):
        # uniform initialization functions
        self.machine = Machine(model=self, states=LeakyBot.states, transitions=LeakyBot.transitions, initial='waiting')
        self.direction = 'fwd' 
        self.probability = np.zeros((6,6)) 
        
        # attach motors to object
        self.motor_left = motor_left
        self.motor_right = motor_right
        #self.get_graph().draw('state_diagram.png', prog='dot')
        
        self.driving_clock = 0
        self.sensing_clock = 0
        
        self.cam_flag = 1
        self.speed = 0
        self.high_humidity = True
        self.have_block = False

    def do_have_block(self):
        return(self.have_block)

    def do_high_humidity(self):
        return(self.high_humidity)

    def set_probability(self, diagvec):
        self.probability = np.diagflat([diagvec])

    def update_probability(self, add_vec):
        prob_matrix = self.probability. astype(float)
        prob_vec = np.diagonal(prob_matrix).copy()
        #print("debug: ", prob_vec, add_vec)
        prob_vec += add_vec.astype(float)
        ret_matrix = np.diagflat([prob_vec])

        self.probability = ret_matrix

    def on_enter_turning(self):
        print(self.state)

    def on_exit_turning(self):
        # Pause briefly before transition
        self.auto_set_motor_values(0,0)
        time.sleep(2)

    def on_enter_driving(self):
        print(self.state)
        self.direction = 'fwd'
        self.auto_set_motor_values(self.speed, self.speed)
        self.driving_clock = time.time()

    def on_exit_driving(self):
        self.auto_set_motor_values(0,0)
        

    def on_enter_sensing(self):
        print(self.state)
        # control the motors here.
        self.auto_set_motor_values(0,0)
        self.sensing_clock = time.time()
        self.sensor_loop += 1

    # no need for this?
    def set_loop(self):
        self.sensor_loop = 0

    def set_turn_direction(self):
        # alternates directions to a) avoid dripper jam and b) hopefully keep block in place
        if self.direction == 'turn': self.direction = 'revturn'
        elif self.direction == 'revturn': self.direction = 'turn'

    def set_turn_status(self):
        turn_dir = bool(random.getrandbits(1))
        self.direction = 'revturn'
        if turn_dir: self.cam_flag = 1
        else: self.cam_flag = 0
             
        print("Turning direction: ", self.cam_flag)
        self.auto_set_motor_values(0,0)
        time.sleep(1.5)

    def go_backwards(self, drive_time):
        store_direction = self.direction
        self.direction = 'rev'
        self.auto_set_motor_values(self.speed, self.speed)
        time.sleep(drive_time)
        self.auto_set_motor_values(0,0)
        self.direction = store_direction
    
    def generic_turn(self, drivetime):
        store_direction = self.direction
        self.direction = 'turn'
        self.auto_set_motor_values(self.speed, self.speed)
        time.sleep(drivetime)
        self.auto_set_motor_values(0,0)
        self.direction = store_direction
        
    def generic_rev_turn(self, drivetime):
        store_direction = self.direction
        self.direction = 'revturn'
        self.auto_set_motor_values(self.speed, self.speed)
        time.sleep(drivetime)
        self.auto_set_motor_values(0,0)
        self.direction = store_direction
    
    def on_enter_waiting(self):
        print(self.state)
        self.auto_set_motor_values(0,0)
        
    
    def on_exit_waiting(self):
        time.sleep(2)
        self.have_block = True
        self.high_humidity = True
        self.start_turning_frame = 0
        
        turn_dir = bool(random.getrandbits(1))
        if turn_dir:
            self.cam_flag = 1
        else:
            self.cam_flag = 0

        self.direction = 'turn'

        
    def on_enter_deposit(self):
        print(self.state)
        self.direction = 'fwd'
        
    def on_exit_deposit(self):
        self.have_block = False

        self.auto_set_motor_values(0,0)
        time.sleep(1.5)
            
        
    def on_enter_backup(self):
        print(self.state)
        self.direction = 'rev'
        self.auto_set_motor_values(self.speed, self.speed)
        time.sleep(0.6)
        # different to a normal turn, so:

        self.direction = 'revturn'
        self.auto_set_motor_values(0,0)

            
    def on_exit_backup(self):
        self.direction = 'revturn' # this is redundant, surely?
        self.auto_set_motor_values(0,0)
        time.sleep(2)
        
  
    def on_enter_go_home(self):
        print("Now entering:" , self.state)
        self.high_humidity = True # probably not necessary but anyway


    def set_motor_values(self, left_speed, right_speed, left_dir, right_dir):
		# more directly controllable version of set motor values, requires
		# knowledge of robot state as input - use when heading direction is
		# fixed and known
		ml = self.motor_left
		mr = self.motor_right
		ml.setSpeed(left_speed)
		mr.setSpeed(right_speed)
		ml.run(left_dir)
		mr.run(right_dir)
		
  
    def auto_set_motor_values(self, left_speed, right_speed):
		# checks direction and sets eveyrthing automatically
        ml = self.motor_left
        mr = self.motor_right
        check_dir = self.direction
        
        #ml.setSpeed(int(float(left_speed)*1.2))
        ml.setSpeed(left_speed)
        mr.setSpeed(right_speed)

        if (left_speed == 0) and (right_speed == 0):
            ml.run(Adafruit_MotorHAT.RELEASE)
            mr.run(Adafruit_MotorHAT.RELEASE)

        elif check_dir == 'fwd':
            #print("forward check")
            ml.run(Adafruit_MotorHAT.FORWARD)
            mr.run(Adafruit_MotorHAT.FORWARD)
        
        elif check_dir == 'rev':
            ml.run(Adafruit_MotorHAT.BACKWARD)
            mr.run(Adafruit_MotorHAT.BACKWARD)
        
        elif check_dir == 'revturn':           
           if self.cam_flag:
               ml.setSpeed(int(float(right_speed)*1.2))
               mr.setSpeed(int(float(right_speed)*0.3))

           else:
               mr.setSpeed(int(float(left_speed)*1.2))
               ml.setSpeed(int(float(left_speed)*0.3))

           ml.run(Adafruit_MotorHAT.BACKWARD)
           mr.run(Adafruit_MotorHAT.BACKWARD)

        elif check_dir == 'turn':
           if self.cam_flag:
               ml.setSpeed(int(float(right_speed)*0.3))
               mr.setSpeed(int(float(right_speed)*1.2))
           else:
               mr.setSpeed(int(float(left_speed)*0.3))
               ml.setSpeed(int(float(left_speed)*1.2))

           ml.run(Adafruit_MotorHAT.FORWARD)
           mr.run(Adafruit_MotorHAT.FORWARD)

        #elif check_dir == 'left':
           
        #    ml.run(Adafruit_MotorHAT.BACKWARD)
        #    mr.run(Adafruit_MotorHAT.FORWARD)

        #elif check_dir == 'right':
        #    ml.run(Adafruit_MotorHAT.FORWARD)
        #    mr.run(Adafruit_MotorHAT.BACKWARD)
            
        else: 
            ml.run(Adafruit_MotorHAT.FORWARD)
            mr.run(Adafruit_MotorHAT.FORWARD)
            
        
            
        
    def have_block(self):
        print(self.have_block)
        return self.have_block
        
    def high_humidity(self):
        return self.high_humidity
        
        

