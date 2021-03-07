import pyrealsense2 as rs
import numpy.core.multiarray
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import time
import sys
import os
import pandas as pd


# Methode zum Umwandlen der Bildkoordinaten in 3D-Koordinaten
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    # Auslesen der intrinsischen Parameter
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.ppx
    _intrinsics.ppy = cameraInfo.ppy
    _intrinsics.fx = cameraInfo.fx
    _intrinsics.fy = cameraInfo.fy
    _intrinsics.model = cameraInfo.model
    _intrinsics.coeffs = cameraInfo.coeffs

    # Berechnen der Koordinaten und Speichern in Array
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)

    # result[0]: right, result[1]: down, result[2]: forward
    return result[0], -result[1], -result[2]

# Callback-Methode die aufgerufen wird, sobald das Schachbrett mit Doppelklick angeklickt wurde
# Sucht den nächsten Punkt auf dem Schachbrett und speichert Werte, wenn Tiefenwert definiert ist
# Bekommt die Koordinaten des Klicks übergeben
def checkboard_clicked(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:

        # Global genutzte Variablen deklarieren
        global corners2
        global point_count
        global camera_coords
        global world_x
        global world_y
        global world_z
        global run
        global detected
        global camera_x
        global camera_y
        global camera_z

        # Erzeugen der Variablen für die kürzesten Distanzen
        shortest = 1000
        x_shortest = 1000
        y_shortest = 1000

        # Durchlaufen der erkannten Ecken
        for corner in corners2:
            # Koordinaten aus Ecken auslesen und Distanz berechnen
            x_c,y_c = corner.ravel()
            distance = math.sqrt((x_c - x)**2 + (y_c - y)**2)

            # Prüfen, ob Ecke am nächsten am Klick ist und ggf. Werte zwischenspeichern
            if distance < shortest:
                shortest = distance
                x_shortest = x_c
                y_shortest = y_c

        # Distanz zum nächsten Punkt am Klick berechnen
        dist = aligned_depth_frame.get_distance(x_shortest, y_shortest)
        # Auslesen der intrinsischen Parameter und berechnen der Kamerakoordinaten
        camera_info = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        x_w, y_w, z_w = convert_depth_to_phys_coord_using_realsense(x, y, dist, camera_info)

        # Prüfen, ob alle Punkte ungleich 0 sind
        if x_w != 0 and y_w != 0 and z_w != 0:

            # Erkannte Ecke auf dem Bild markieren
            cv2.circle(detected, (x_shortest, y_shortest), 2, (255,255,255), thickness=2)
            cv2.imshow('Calibration', detected)

            # Speichern der Kamerakoordinaten für diesen Punkt
            camera_coords.append([float(x_w), float(y_w), float(z_w)]) 
            camera_x.append(float(x_w))
            camera_y.append(float(y_w))   
            camera_z.append(float(z_w))              

            # Abfragen der realen Koordinaten für den gleichen Punkt und Speichern dieser
            x_platte = (float(input("X-Wert bis zur Platte in m: ")))
            corner_spalte = (int(input("Spaltennummer der Ecke (ganz links = 0): ")))
            corner_zeile = (int(input("Zeilennummer der Ecke (ganz unten = 0): ")))

            world_x_temp = x_platte + 0.085 + corner_spalte * 0.055 + corner_zeile * 0.000167
            world_y_temp = 0.085 + corner_zeile * 0.055167 - corner_spalte * 0.000167
            world_z_temp = float(input("Tiefe der Ebene in m: "))

            world_x.append(world_x_temp)
            world_y.append(world_y_temp)
            world_z.append(world_z_temp)

            # Anzahl der aufgenommenen Punkte inkrementieren
            point_count = point_count + 1
        else:
            # Markieren, dass für den gewählten Punkt kein Tiefenwert existiert
            cv2.circle(detected, (x_shortest, y_shortest), 2, (0,0,0), thickness=2)
            cv2.imshow('Calibration', detected)

        # Prüfen, ob genug Punkte aufgenommen wurden und ggf. Matrix berechnen
        if point_count == MAX_POINTS:
            calculate_equation()
            run = False

# Wird aufgerufen, wenn die Anzahl der aufzunehmenden Punkte erreicht ist
# Berechnet die Transformationsmatrix durch lineare Regression und gibt diese aus
def calculate_equation():

    # Nutzen der globalen Arrays mit Koordinaten
    global camera_coords
    global world_x
    global world_y
    global world_z
    global camera_x
    global camera_y
    global camera_z

    # Coordinaten in Array sammeln um sie als Array in json-file zu speichern
    json_array_camera = []
    json_array_camera.append(camera_x)
    json_array_camera.append(camera_y)
    json_array_camera.append(camera_z)
    json_array_world = []
    json_array_world.append(world_x)
    json_array_world.append(world_y)
    json_array_world.append(world_z)

    df = pd.DataFrame({"camera" : json_array_camera, "world" : json_array_world})
    df.to_json("Kalibrierung.json")

    # Umwandeln der Matritzen für die lineare Regression
    x_matr = np.array(camera_coords)
    y1_matr = np.array(world_x)
    y2_matr = np.array(world_y)
    y3_matr = np.array(world_z)

    # Ermitteln der ersten Zeile mit Kamerakoordinaten und x-Werten der realen Koordinaten
    model_1 = linear_model.LinearRegression()
    model_1.fit(x_matr, y1_matr)

    # Ermitteln der zweiten Zeile mit Kamerakoordinaten und y-Werten der realen Koordinaten
    model_2 = linear_model.LinearRegression()
    model_2.fit(x_matr, y2_matr)

    # Ermitteln der dritten Zeile mit Kamerakoordinaten und z-Werten der realen Koordinaten
    model_3 = linear_model.LinearRegression()
    model_3.fit(x_matr, y3_matr)

    # Ausgeben der Zeilen für die Rotationsmatrix
    print("Zeile 1: " + str(model_1.coef_))
    print("Zeile 2: " + str(model_2.coef_))
    print("Zeile 3: " + str(model_3.coef_))

    # Ausgeben der Intercept-Werte für den Translationsvektor der Matrix
    print("Intercept X: " + str(model_1.intercept_))
    print("Intercept Y: " + str(model_2.intercept_))
    print("Intercept Z: " + str(model_3.intercept_))

    return

# Setzen der Anzahl an aufzunehmenden Punkte
MAX_POINTS = 100

# Variablen deklarieren, die später global nutzbar sein sollen
camera_coords = []
world_x = []
world_y = []
world_z = []
camera_x = []
camera_y = []
camera_z = []
point_count = 0
run = True

# Setzen der Reihen und Spalten des Schachbretts
rows = 7
columns = 7

# Starten des Videostreams der Tiefenkamera
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(cfg)

# Einstellen des 'High Accuracy' Modus
depth_sensor = profile.get_device().first_depth_sensor()
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print('%02d: %s'%(i,visulpreset))
    if visulpreset == "High Accuracy":
        depth_sensor.set_option(rs.option.visual_preset, i)
# Erhöhen der Laserpower des Emitters
depth_sensor.set_option(rs.option.laser_power, 210)
# Veringern der kleinsten Tiefeneinheit für genauere Ergebnisse
depth_sensor.set_option(rs.option.depth_units, 0.0005)

# Die ersten Frames überspringen, damit die Belichtung sich einstellen kann
for x in range(10):
    pipe.wait_for_frames()

# Schleife für das Auslesen und Erkennen des Schachbretts
while(run):

    # Aktuelle frames werden gespeichert
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Überprüfen, ob sowohl Tiefen- als auch Farbbild vorhanden sind
    if color_frame and depth_frame:

        # Die beiden Frames der Kamera zueinander ausrichten
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Neuen Farbframe holen und für Weiterverarbeitung als numpy-Array speichern
        color_frame = frameset.get_color_frame()
        color = np.asanyarray(color_frame.get_data())

        # Tiefenwerte mit Hilfe des colorizers farbig darstellen und als numpy-Array speichern
        colorizer = rs.colorizer()
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # criteria für das Erkennen der Ecken erzeugen
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Bild als Grauwertbild speichern
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # Finden der Ecken des Schachbretts
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        # Prüfen, ob ein Schachbrett erkannt wurde
        if ret:

            # Erkannte Ecken auslesen und speichern
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            detected = cv2.drawChessboardCorners(gray, (rows, columns), corners2, ret)
            
            # Grauwertbild für die Kalibrierung zeigen und ClickListener hinzufügen
            cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Calibration', detected)
            cv2.setMouseCallback('Calibration', checkboard_clicked)
            # Warten auf Tastendruck
            # So kann überprüft werden, ob die Ecken auch wirklich richtig sitzen und erst das nächste Bild angezeigt werden, wenn Punkte auf der Ebene eingegeben wurden
            cv2.waitKey()
    else:
        print("Image is null")

# Streams der Kamera stoppen
pipe.stop()