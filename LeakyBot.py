from transitions import Machine
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor

# could extend this by bringing in motor functions (lowpass filter, etc)

class LeakyBot(object):
    # each leaky bot should have two motors, left and right, 
    # and a standard set of states and transitions
    # Conditions may be unecessary and slowing things down
    
    states = ['waiting', 'turning', 'sensing', 'deposit', 'backup', 'go_home']
    
    transitions = [ {'trigger':'button_push', 'source':'waiting', 'dest':'turning'},
        {'trigger':'walls_balanced', 'source':'turning', 'dest':'sensing', 'conditions':['do_have_block', 'do_high_humidity']}, 
        {'trigger':'low_humidity', 'source':'sensing', 'dest':'turning'},
        {'trigger':'wall_found', 'source':'turning', 'dest':'deposit', 'conditions':'do_have_block'},
        {'trigger':'reached_wall', 'source':'deposit', 'dest':'backup'},
        {'trigger':'home_spotted', 'source':'backup', 'dest':'go_home', 'unless':'do_have_block'},
        {'trigger':'close_to_home', 'source':'go_home', 'dest':'waiting'} ]
        
        
    high_humidity = True
    have_block = False
    speed = 0
    threshold_state = 'leq'
    deposit_direction = 'left'
    
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
        des_speed = self.speed
        self.set_motor_values(des_speed)
        
    
    def on_enter_waiting(self):
        self.set_motor_values(0)
        
    def on_enter_deposit(self):
        self.direction = 'fwd' 
        des_speed = self.speed
        self.set_motor_values(des_speed)
        
    def on_enter_backup(self):
        self.direction = 'rev'
        ml = self.motor_left
        mr = self.motor_right

        des_speed = self.speed
        if self.deposit_direction == 'left':
            ml.setSpeed(des_speed)
            mr.setSpeed(0)
            
        else:
            ml.setSpeed(0)
            mr.setSpeed(des_speed)

        ml.run(Adafruit_MotorHAT.BACKWARD)
        mr.run(Adafruit_MotorHAT.BACKWARD)
            
        
  
    def on_enter_go_home(self):
        self.direction = 'fwd'
        self.high_humidity = True
        des_speed = self.speed
        self.set_motor_values(des_speed)
        
        
    def on_enter_close_to_home(self):
        self.set_motor_values(0)
  
    def set_motor_values(self, *arg):
        ml = self.motor_left
        mr = self.motor_right

        if len(arg) == 2:
            # differential drive with specific values
            # does changing the speed require a motor reset? shouldn't
            # todo: change so we can have a differential drive in either direction
            # then clean up eg. backup.
            ml.setSpeed(arg[0])
            mr.setSpeed(arg[1])
            
        else:
            # we are setting a default speed for any maneouvers.
            # it might actually be better code to force the user to use two arguments, but anyway
            check_dir = self.direction
            speed = arg[0]
            
            # if speed = 0, stop
            if speed == 0:
                ml.run(Adafruit_MotorHAT.RELEASE)
                mr.run(Adafruit_MotorHAT.RELEASE)

            elif check_dir == 'rev':
                ml.setSpeed(speed)
                mr.setSpeed(speed)
        
                ml.run(Adafruit_MotorHAT.BACKWARD)
                mr.run(Adafruit_MotorHAT.BACKWARD)
            
            else:            
                if check_dir == 'left':
                    ml.setSpeed(0)
                    mr.setSpeed(speed)
            
                elif check_dir == 'right':
                    ml.setSpeed(speed)
                    mr.setSpeed(0)
                
                elif check_dir == 'fwd':
                    ml.setSpeed(speed)
                    mr.setSpeed(speed)
                
                else:
                    print("Warning: direction not recognized")
        
                ml.run(Adafruit_MotorHAT.FORWARD)
                mr.run(Adafruit_MotorHAT.FORWARD)
            
          
        
    def have_block(self):
        return self.have_block
        
    def high_humidity(self):
        return self.high_humidity
        
        

