import pigpio
import subprocess

#initialize pigpio if it was not previously initialized
out = subprocess.getoutput('sudo pigpiod')

pi = pigpio.pi()

class Servo_motor:
    

    def __init__(self, port):
        pi.set_mode(port, pigpio.OUTPUT)
        self.port = port
        
    #Sets angle of servo(0-1000)
    def set_angle(self, angle):
        if angle > 1000: angle = 1000
        if angle < 0: angle = 0

        pulsewidth_angle = (angle * 2) + 500
        pi.set_servo_pulsewidth(self.port, pulsewidth_angle)

        return 1
                
    #reads angle of servo(0-1000)
    def get_angle(self):
        angle = pi.get_servo_pulsewidth(self.port)
        actual_angle = (angle - 500) / 2
        return actual_angle
    