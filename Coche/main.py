import socket
import struct
import pickle
import cv2
#import artemis_autonomous_car

import time
from pynput import keyboard
import numpy as np
import cmath, math

import torch
import ultralytics
import threading

from Funciones import *
from Argumentos import *

server_address = ('10.0.128.174',20001)

control_acelerador=0
control_giro=0
datos = None

def on_press(key):
    global control_acelerador
    global control_giro
    if key==keyboard.KeyCode.from_char('2'):
        control_acelerador=1
    if key==keyboard.KeyCode.from_char('w'):
        control_acelerador=0.6
    if key==keyboard.KeyCode.from_char('s'):
        control_acelerador=-0.5
    if key==keyboard.KeyCode.from_char('x'):
        control_acelerador=-0.9
    if key==keyboard.KeyCode.from_char('a'):
        control_giro=1
    if key==keyboard.KeyCode.from_char('d'):
        control_giro=-1
    if key==keyboard.KeyCode.from_char('+'):
        numFrames += 1
        frame = 0
    if key==keyboard.KeyCode.from_char('-'):
        numFrames -= 1
        frame = 0

def on_release(key):
    global control_acelerador
    global control_giro
    if key==keyboard.KeyCode.from_char('2'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('w'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('s'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('x'):
        control_acelerador=0
    if key==keyboard.KeyCode.from_char('a'):
        control_giro=0.25
    if key==keyboard.KeyCode.from_char('d'):
        control_giro=0.25

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def send_control(control_giro,control_acelerador,address):
    sock.sendto(struct.pack('c',bytes('C','ascii'))+struct.pack('d',round(control_giro,3))+struct.pack('d',round(control_acelerador,3)),address)

def send_request_thread(modelo, traqueo, image_bytes):
    global datos
    datos = send_request(file_list=[np.frombuffer(image_bytes, dtype=np.uint8)], model_name = modelo, tracking = traqueo)
    
def local_thread(modelo, image_bytes):
    global datos
    datos = local(imagen=[np.frombuffer(image_bytes, dtype=np.uint8)], model_name = modelo)
    print("Si que entra\n", datos)

if __name__ == "__main__":
    
    args = parse_opt()

    # Si se proporciona el argumento -help, imprime la ayuda y finaliza
    if args.help:
        print_help()
        exit()
        
    if args.modelName:
        model_name = args.modelName
    else:
        print("\nNo has seleccionado el modelo\nEscribe python3 main.py -modelName <modelo>")
        exit()
        
    if args.tracking:
        traqueo = True
    else:
        traqueo = False

    firstLaunch = 1
    muestreo = args.muestreo
    conteo = muestreo
    received_payload=b''
    
    
    if not args.webcam:
        #auto_utils=artemis_autonomous_car.artemis_autonomous_car([0])
        #Generacion de socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sock.bind(server_address)
    
        #try:
            # Receive the data in small chunks and retransmit it
        while True:
            #print("Esperando dato")
            data, address = sock.recvfrom(99999)
            #print("Dato recibido")
            data_type = struct.unpack('c',bytes([data[0]]))[0]
            #print("Tipo de dato: ",data_type)
            received_payload=bytes(data[1:])
            data = pickle.loads(received_payload,encoding='latin1')
            if data_type == b'I':
                img=cv2.imdecode(data,1)
                img = cv2.resize(img, None, None, fx=1.5, fy=1.5)
                if conteo == muestreo:
                    # Convert the image to bytes
                    ret, buffer = cv2.imencode('.jpg', img)
                    image_bytes = np.array(buffer).tobytes()
                    if args.cloud:
                        if firstLaunch == 1:
                            thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                            thread.start()
                            firstLaunch = 0
                        elif not thread.is_alive():
                            thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                            thread.start()
                    else:
                        if firstLaunch == 1:
                            thread = threading.Thread(target=local_thread, args=(model_name, image_bytes,))
                            thread.start()
                            firstLaunch = 0
                        elif not thread.is_alive():
                            thread = threading.Thread(target=local_thread, args=(model_name, image_bytes,))
                            thread.start()  

                    conteo = 0
                else:
                    conteo += 1

                imagen = dibujar_caja(img, datos)
                cv2.imshow("Coche ARTEMIS", imagen)
                cv2.waitKey(1)
                send_control(control_giro,control_acelerador,address)
            if data_type == b'D':
                pass
            if data_type == b'L':
                img_lidar = np.zeros((480,640,3), np.uint8)
                i=0
                ranges=data
                for range in ranges:
                    Z=cmath.rect(range,math.radians(-i-90))
                    i = i+1
                    if Z.real != float('inf') and Z.real != float('-inf'):
                        x=int(Z.real*70)+320
                    else:
                        x=0
                    if Z.imag != float('inf') and Z.imag != float('-inf'):
                        y=int(Z.imag*70)+240
                    else:
                        y=0
                    cv2.circle(img_lidar,(x,y), radius=2, color=(0,0,255),thickness=2)
                    
                cv2.imshow("Mapa LIDAR",img_lidar)
                cv2.waitKey(1)
                pass
        #except Exception:
        #    pass   
        #finally:
            # Clean up the connection
        #    pass#connection.close()
    else:
        # Crear la ventana donde se mostrará el video
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        
        cap = cv2.VideoCapture(0)
        while cap.isOpened(): # Si nuestra webcam esta activa
            ret, frame = cap.read() # En frame se encuentra la imagen
            
            frame = cv2.resize(frame, None, None, fx=1.5, fy=1.5)
                
            if conteo == muestreo:
                # Convert the image to bytes
                ret, buffer = cv2.imencode('.jpg', frame)
                image_bytes = np.array(buffer).tobytes()
                
                if args.cloud:
                    if firstLaunch == 1:
                        thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                        thread.start()
                        firstLaunch = 0
                    elif not thread.is_alive():
                        thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                        thread.start()
                else:
                    if firstLaunch == 1:
                        thread = threading.Thread(target=local_thread, args=(model_name, image_bytes,))
                        thread.start()
                        firstLaunch = 0
                    elif not thread.is_alive():
                        thread = threading.Thread(target=local_thread, args=(model_name, image_bytes,))
                        thread.start()  

                conteo = 0
            else:
                conteo += 1
                
            imagen = dibujar_caja(frame, datos)
            cv2.imshow("Video", imagen)

            # Esperar un poco antes de mostrar la siguiente imagen
            cv2.waitKey(1)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Para parar se le da a la 'q'
                break
        cap.release()
        cv2.destroyAllWindows()
