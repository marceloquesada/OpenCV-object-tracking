import pigpio
import os


#cria controle de gpio
pi = pigpio.pi()

#tenta inicializar o pigpio
try:
    os.system("sudo pigpiod")
except:pass


#classe para controlar os servos
class Servo_motor:
    
    #no init o usuário deve fornacer a porta, que então será setada e salva em self.port   
    def __init__(self, port):
        pi.set_mode(port, pigpio.OUTPUT)
        self.port = port
        
    #essa função seta o ângulo do servo (0-1000)
    def set_angle(self, angle):
        pulsewidth_angle = (angle * 2) + 500
        pi.set_servo_pulsewidth(self.port, pulsewidth_angle)
                
    #essa função lê o ângulo do servo(0-1000)
    def get_angle(self):
        angle = pi.get_servo_pulsewidth(self.port)
        actual_angle = (angle - 500) / 2
        
        return angle
    
