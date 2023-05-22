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

from sort import Sort

###VARIABLES GLOBALES###
server_address = ('10.0.128.174',20001)

control_acelerador=0
control_giro=0

datos = None

do = threading.Event() #Do processing
lock = threading.Lock()
imagen_procesar = None
traqueo = None
#######################

###CONTROL DE COCHE###
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
###############################

###PROCESADO IMAGEN###
#REMOTO
def send_request_thread(modelo, traqueo, image_bytes):
    global datos
    local_data = send_request(file_list=[np.frombuffer(image_bytes, dtype=np.uint8)], model_name = modelo, tracking = traqueo)

    lock.acquire()
    datos = local_data
    lock.release()

#LOCAL
def procesado_imagen():
    # modelo_clasificador = torch.hub.load('ultralytics/yolov5', 'custom', path='./modelos/smallFinal.pt', force_reload=True)
    modelo_clasificador = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    global imagen_procesar
    global datos

    # Crear la instancia de SORT
    mt_tracker = Sort()

    while True:
        if do.is_set():
            img = imagen_procesar
            results = modelo_clasificador(img)
            local_data = results_to_json(results, modelo_clasificador)
            if traqueo:
                detections = results.pred[0].numpy()
                # Actualizar SORT
                track_bbs_ids = mt_tracker.update(detections)
                
                if len(track_bbs_ids) > 0:
                    for j in range(len(track_bbs_ids.tolist())):
                        coords = track_bbs_ids.tolist()[j]
                        
                        # Agregar el ID actualizado a json_results
                        local_data[0][j]['tracker_id'] = int(coords[4])

            lock.acquire()
            datos = local_data
            lock.release()
            do.clear()
        else:
            time.sleep(1)
#######################

###MAIN###
if __name__ == "__main__":
    args = parse_opt()

    muestreo = args.muestreo
    conteo = muestreo
    received_payload=b''

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

    if not args.cloud:
        do.clear()
        thread = threading.Thread(target=procesado_imagen, args=())
        thread.start()

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
                #img = cv2.resize(img, None, None, fx=1.5, fy=1.5) DANGER
                if conteo == muestreo:
                    if args.cloud:
                        # Convert the image to bytes
                        ret, buffer = cv2.imencode('.jpg', img)
                        image_bytes = np.array(buffer).tobytes()

                        try:
                            if not thread.is_alive():
                                thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                                thread.start()
                        except:
                            thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                            thread.start()

                    elif not do.is_set():
                        imagen_procesar = img
                        do.set()

                    conteo = 0
                else:
                    conteo += 1

                lock.acquire()
                imagen = dibujar_caja(img, datos)
                lock.release()

                imagen = cv2.resize(imagen, None, None, fx=1.5, fy=1.5)
                cv2.imshow("Coche ARTEMIS", imagen)
                cv2.waitKey(1)
                send_control(control_giro,control_acelerador,address)

            if data_type == b'D':
                pass
            if data_type == b'L':
                pass

    else:
        # Crear la ventana donde se mostrará el video
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

        cap = cv2.VideoCapture(0)
        while cap.isOpened(): # Si nuestra webcam esta activa
            ret, frame = cap.read() # En frame se encuentra la imagen

            img = cv2.resize(frame, None, None, fx=1.5, fy=1.5)

            if conteo == muestreo:
                    if args.cloud:
                        # Convert the image to bytes
                        ret, buffer = cv2.imencode('.jpg', img)
                        image_bytes = np.array(buffer).tobytes()

                        try:
                            if not thread.is_alive():
                                thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                                thread.start()
                        except:
                            thread = threading.Thread(target=send_request_thread, args=(model_name, traqueo, image_bytes,))
                            thread.start()
                    elif not do.is_set():
                        imagen_procesar = img
                        do.set()

                    conteo = 0
            else:
                conteo += 1

            lock.acquire()
            imagen = dibujar_caja(img, datos)
            lock.release()
            cv2.imshow("Video", imagen)

            # Esperar un poco antes de mostrar la siguiente imagen
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Para parar se le da a la 'q'
                break
        cap.release()
        cv2.destroyAllWindows()
