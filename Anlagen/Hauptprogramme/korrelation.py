import numpy.core.multiarray
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import pandas as pd
import ast
import re, ast

# Auslesen der Kamera- und Weltkoordinaten aus json-File
df = pd.read_json("Kalibrierung.json")
camera_coords = df['camera'].to_numpy()
world_coords = df['world'].to_numpy()

# Initialisieren der Arrays zum Speichern der Kamer- und Weltkoordinaten
coord_array_camera = []
coord_array_world = []

# Ausgelesene Listen in Numpy-Array umwandeln
# Kamerakoordinaten
for coord in camera_coords:
    temp = np.array(coord)
    coord_array_camera.append(temp)
coord_array_camera_np = np.array(coord_array_camera)
# Weltkoordinaten
for coord in world_coords:
    temp = np.array(coord)
    coord_array_world.append(temp)
coord_array_world_np = np.array(coord_array_world)

# Speichern der Koordinaten in Punktvektoren und Normieren
camera_vector = []
camera_vector_manual = []
camera_vector_normalized = []
world_x_norm = []
world_y_norm = []
world_z_norm = []
for i in range(len(coord_array_camera_np[0])):
    # Kamerakoordinaten
    x = float(coord_array_camera_np[0][i])
    y = float(coord_array_camera_np[1][i])
    z = float(coord_array_camera_np[2][i])
    # Normieren der Kamerakoordinaten
    x_norm = (x - np.mean(coord_array_camera_np[0])) / np.std(coord_array_camera_np[0])
    y_norm = (y - np.mean(coord_array_camera_np[1])) / np.std(coord_array_camera_np[1])
    z_norm = (z - np.mean(coord_array_camera_np[2])) / np.std(coord_array_camera_np[2])
    # Speichern der Daten in Vektoren
    camera_vector.append([float(x), float(y), float(z)])
    camera_vector_manual.append([float(x), float(y), float(z), float(1)])
    camera_vector_normalized.append([float(x_norm), float(y_norm), float(z_norm), float(1)])

    # Weltkoordinaten
    x = float(coord_array_world_np[0][i])
    y = float(coord_array_world_np[1][i])
    z = float(coord_array_world_np[2][i])
    x_norm_w = (x - np.mean(coord_array_world_np[0])) / np.std(coord_array_world_np[0])
    y_norm_w = (y - np.mean(coord_array_world_np[1])) / np.std(coord_array_world_np[1])
    z_norm_w = (z - np.mean(coord_array_world_np[2])) / np.std(coord_array_world_np[2])
    world_x_norm.append(x_norm_w)
    world_y_norm.append(y_norm_w)
    world_z_norm.append(z_norm_w)

# Umwandeln in Numpy-Arrays
world_x_norm_np = np.array(world_x_norm)
world_y_norm_np = np.array(world_y_norm)
world_z_norm_np = np.array(world_z_norm)

# Berechnen der Koeffizientenmatrix
corr = np.corrcoef(coord_array_camera_np)

# Durchführen der linearen Regressionen
# Zeile 1 mit Rohdaten
model_1_raw = linear_model.LinearRegression()
model_1_raw.fit(camera_vector, coord_array_world_np[0])
# Zeile 2 mit Rohdaten
model_2_raw = linear_model.LinearRegression()
model_2_raw.fit(camera_vector, coord_array_world_np[1])
# Zeile 3 mit Rohdaten
model_3_raw = linear_model.LinearRegression()
model_3_raw.fit(camera_vector, coord_array_world_np[2])
# Zeile 1 mit normalisierten Daten
model_1_norm = linear_model.LinearRegression(fit_intercept=False)
model_1_norm.fit(camera_vector_normalized, world_x_norm_np)
# Zeile 2 mit normalisierten Daten
model_2_norm = linear_model.LinearRegression(fit_intercept=False)
model_2_norm.fit(camera_vector_normalized, world_y_norm_np)
# Zeile 3 mit normalisierten Daten
model_3_norm = linear_model.LinearRegression(fit_intercept=False)
model_3_norm.fit(camera_vector_normalized, world_z_norm_np)

# Regression 'per Hand'
linreg_manual = []
A = np.array(camera_vector_manual)
A_T = A.T
c = A_T.dot(A)
d = np.linalg.matrix_power(c, -1)
e = d.dot(A_T)
v_1 = e.dot(coord_array_world_np[0])
v_2 = e.dot(coord_array_world_np[1])
v_3 = e.dot(coord_array_world_np[2])
linreg_manual.append(v_1)
linreg_manual.append(v_2)
linreg_manual.append(v_3)
linreg_manual_np = np.array(linreg_manual)

# Ausgeben der Koeffizientenmatrix
print("Korrelationsmatrix: ")
print(str(corr))
print()
print("_____________________________________________________")
print()
# Ausgabe der ersten Transformationsmatrix
print("Ergebnis der Matrix mit nicht normierten Werten")
print("Transform 1: " + str(model_1_raw.coef_))
print("Transform 2: " + str(model_2_raw.coef_))
print("Transform 3: " + str(model_3_raw.coef_))
print("Intercept X: " + str(model_1_raw.intercept_))
print("Intercept Y: " + str(model_2_raw.intercept_))
print("Intercept Z: " + str(model_3_raw.intercept_))
print()
print("_____________________________________________________")
print()
# Ausgabe der zweiten Transformationsmatrix
print("Ergebnis der Matrix mit normierten Werten")
print("Transform 1: " + str(model_1_norm.coef_))
print("Transform 2: " + str(model_2_norm.coef_))
print("Transform 3: " + str(model_3_norm.coef_))
print("Intercept X: " + str(model_1_norm.intercept_))
print("Intercept Y: " + str(model_2_norm.intercept_))
print("Intercept Z: " + str(model_3_norm.intercept_))
print()
print("_____________________________________________________")
print()
# Ausgabe der dritten Transformationsmatrix
print("Ergebnis der ohne Scikit berechneten Transformationsmatrix")
print(linreg_manual_np)