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

datos = [[],""]
ptime = 0
fps_count = 0

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
    global ptime
    global fps_count
    
    tic = time.perf_counter()
    local_data = send_request(file_list=[np.frombuffer(image_bytes, dtype=np.uint8)], model_name = modelo, tracking = traqueo)
    toc = time.perf_counter()

    lock.acquire()
    datos = local_data
    fps_count += 1
    ptime =  toc-tic
    lock.release()

#LOCAL
def procesado_imagen(modelo_clasificador):
    global imagen_procesar
    global datos
    global ptime
    global fps_count

    # Crear la instancia de SORT
    mt_tracker = Sort()

    while True:
        if do.is_set():
            #print("Im HERE")
            tic = time.perf_counter()
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
                        
            toc = time.perf_counter()

            lock.acquire()
            datos = local_data
            fps_count += 1
            ptime = toc-tic
            
            lock.release()
            do.clear()
        #else:
        #    time.sleep(1)
#######################

###MAIN###
if __name__ == "__main__":
    args = parse_opt()
    muestreo = int(args.muestreo)
    conteo = muestreo
    FPS = 0
    received_payload=b''
    id_stop = 0
    id_proh = 0
    id_oblig = 0
    id_peligro = 0
    aceleracion_defecto = 0.46
    marchaAtras_defecto = -0.35
    derecha = -1
    izquierda = 1
    altura = 350

    # Si se proporciona el argumento -help, imprime la ayuda y finaliza
    if args.help:
        print_help()
        exit()

    if args.demo:
        args.modelName = 'Final.onnx'
        args.cloud = True
        args.tracking = True        

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
        modelo_clasificador = torch.hub.load('ultralytics/yolov5', 'custom', path='./modelos/Final.onnx', force_reload=True)
        #modelo_clasificador = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        thread = threading.Thread(target=procesado_imagen, args=(modelo_clasificador,))
        thread.start()

    if not args.webcam:
        #auto_utils=artemis_autonomous_car.artemis_autonomous_car([0])
        #Generacion de socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sock.bind(server_address)

        TIC = time.perf_counter()

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
                        
                        if args.demo:   
                            #print(datos)               
                            if len(datos[0]):                                
                                etiqueta = []
                                for dato in datos[0]:
                                    etiqueta.append(dato['class_name'])
                                
                                if 'STOP' in etiqueta:
                                    control_giro=0
                                    dato = datos[0][etiqueta.index('STOP')]
                                    if dato.get('tracker_id') != None and dato.get('tracker_id')> id_stop:
                                        id_stop = dato['tracker_id']
                                        if(dato.get('bbox')[2] >= altura):
                                            print("\nAltura caja STOP",dato.get('bbox')[2])
                                            control_acelerador=0
                                            send_control(control_giro,control_acelerador,address)
                                            time.sleep(4)
                                            control_acelerador = aceleracion_defecto
                                            control_giro=0 

                                elif 'Obligacion' in etiqueta:
                                    dato = datos[0][etiqueta.index('Obligacion')]
                                    if dato.get('tracker_id') != None and dato.get('tracker_id')> id_oblig:
                                        id_oblig = dato['tracker_id']
                                        if(dato.get('bbox')[2] >= altura and dato.get('confidence')>=0.9):
                                            print("\nAltura caja OBLIG",dato.get('bbox')[2])
                                            print("\tID OBLIG",id_oblig)
                                            control_acelerador = aceleracion_defecto
                                            control_giro=izquierda
                                
                                elif 'Prohibicion' in etiqueta:
                                    control_giro=0
                                    dato = datos[0][etiqueta.index('Prohibicion')]
                                    if dato.get('tracker_id') != None and dato.get('tracker_id')> id_proh:
                                        id_proh = dato['tracker_id']
                                        if(dato.get('bbox')[2] >= altura and dato.get('confidence')>=0.9):
                                            print("\nAltura caja PROH",dato.get('bbox')[2])
                                            print("\tID PROH",id_proh)
                                            control_acelerador = marchaAtras_defecto

                                elif 'Peligro' in etiqueta:
                                    control_giro=0
                                    dato = datos[0][etiqueta.index('Peligro')]
                                    if dato.get('tracker_id') != None and dato.get('tracker_id')> id_peligro:
                                        id_peligro = dato['tracker_id']                                        
                                        if(dato.get('bbox')[2] >= altura and dato.get('confidence')>=0.9):
                                            print("\nAltura caja PELIGRO",dato.get('bbox')[2])
                                            print("\tID PELIGRO",id_peligro)
                                            control_acelerador=-0.6
                                            send_control(control_giro,control_acelerador,address) 
                                            time.sleep(1)
                                            control_acelerador = aceleracion_defecto
                                            control_giro=0

                            else:
                                control_acelerador = aceleracion_defecto
                                control_giro = 0
                            
                            send_control(control_giro,control_acelerador,address)    
                                            
                    elif not do.is_set():
                        imagen_procesar = img
                        do.set()

                    conteo = 0
                else:
                    conteo += 1

                lock.acquire()
                imagen = dibujar_caja(img.copy(), datos)
                timer = ptime
                TOC = time.perf_counter()
                if TOC - TIC > 2:
                    FPS = fps_count/(TOC - TIC)
                    fps_count = 0
                    TIC = time.perf_counter()
                lock.release()

                imagen = cv2.resize(imagen, None, None, fx=1.5, fy=1.5)
                cv2.putText(imagen, "RTT: "+f'{timer*1000:.2f}'+"ms", (20,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
                if len(datos) > 1:
                    cv2.putText(imagen, "T. proc: "+datos[1]+"ms", (20,690), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
                cv2.putText(imagen, "FPS: "+f'{FPS:.2f}', (20,610), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
                cv2.imshow("Coche ARTEMIS", imagen)
                cv2.waitKey(1)
                
                send_control(control_giro,control_acelerador,address)

            if data_type == b'D':
                pass
            if data_type == b'L':
                pass

    else:
        TIC = time.perf_counter()
        
        # Crear la ventana donde se mostrará el video
        cv2.namedWindow("Coche ARTEMIS", cv2.WINDOW_NORMAL)

        cap = cv2.VideoCapture(0)
        while cap.isOpened(): # Si nuestra webcam esta activa
            ret, img = cap.read() # En frame se encuentra la imagen

            # img = cv2.resize(frame, None, None, fx=1.5, fy=1.5)

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
            imagen = dibujar_caja(img.copy(), datos)
            timer = ptime
            TOC = time.perf_counter()
            if TOC - TIC > 2:
                FPS = fps_count/(TOC - TIC)
                fps_count = 0
                TIC = time.perf_counter()
            lock.release()

            imagen = cv2.resize(imagen, None, None, fx=1.5, fy=1.5)
            cv2.putText(imagen, "RTT: "+f'{timer*1000:.2f}'+"ms", (20,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
            if len(datos) > 1:
                cv2.putText(imagen, "T. proc: "+datos[1]+"ms", (20,690), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
            cv2.putText(imagen, "FPS: "+f'{FPS:.2f}', (20,610), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Coche ARTEMIS", imagen)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Para parar se le da a la 'q'
                break
        cap.release()
        cv2.destroyAllWindows()
