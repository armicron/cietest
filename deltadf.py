import numpy as np
import mysql.connector
import pandas as pd



def deltae_cie_2000(lab_color_vector, lab_color_matrix):
    Kl=1
    Kc=1
    Kh=1
    L, a, b = lab_color_vector
    avg_Lp = (L + lab_color_matrix[:, 0]) / 2.0
    C1 = np.sqrt(np.sum(np.power(lab_color_vector[1:], 2)))
    C2 = np.sqrt(np.sum(np.power(lab_color_matrix[:, 1:], 2), axis=1))
    avg_C1_C2 = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(np.power(avg_C1_C2, 7.0) / (np.power(avg_C1_C2, 7.0) + np.power(25.0, 7.0))))
    a1p = (1.0 + G) * a
    a2p = (1.0 + G) * lab_color_matrix[:, 1]
    C1p = np.sqrt(np.power(a1p, 2) + np.power(b, 2))
    C2p = np.sqrt(np.power(a2p, 2) + np.power(lab_color_matrix[:, 2], 2))
    avg_C1p_C2p = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b, a1p))
    h1p += (h1p < 0) * 360
    h2p = np.degrees(np.arctan2(lab_color_matrix[:, 2], a2p))
    h2p += (h2p < 0) * 360
    avg_Hp = (((np.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_Hp)) + \
        0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - \
        0.2 * np.cos(np.radians(4 * avg_Hp - 63))
    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (np.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720
    delta_Lp = lab_color_matrix[:, 0] - L
    delta_Cp = C2p - C1p
    delta_Hp = 2 * np.sqrt(C2p * C1p) * np.sin(np.radians(delta_hp) / 2.0)
    S_L = 1 + ((0.015 * np.power(avg_Lp - 50, 2)) / np.sqrt(20 + np.power(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T
    delta_ro = 30 * np.exp(-(np.power(((avg_Hp - 275) / 25), 2.0)))
    R_C = np.sqrt((np.power(avg_C1p_C2p, 7.0)) / (np.power(avg_C1p_C2p, 7.0) + np.power(25.0, 7.0)))
    R_T = -2 * R_C * np.sin(2 * np.radians(delta_ro))
    return np.sqrt(
        np.power(delta_Lp / (S_L * Kl), 2) +
        np.power(delta_Cp / (S_C * Kc), 2) +
        np.power(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))

def deltas(nums):
    listoflists = []
    for num in range(nums):
        listoflists.append([num])
    return listoflists

def manage(inp):
    for i in inp:
        num = i[0]
        lab1idx = color[num][0]-1
        cie_l1 = np.float(color2[num][0])
        cie_a1 = np.float(color2[num][1])
        cie_b1 = np.float(color2[num][2])
        lab_color_vector = np.array([cie_l1, cie_a1, cie_b1])
        lab_color_matrix = the_matrix
        delta_e = deltae_cie_2000(lab_color_vector, lab_color_matrix)
        df.ix[lab1idx] = delta_e

cnx = mysql.connector.connect(user='user', password='password', host='localhost', database='db_name')
cursor = cnx.cursor()
selectstmt = 'SELECT CIE_L, CIE_a, CIE_b FROM `db_name`.`B2Color` LIMIT 100'
countstmt = 'SELECT colorID, CIE_L, CIE_a, CIE_b FROM `db_name`.`B2Color` LIMIT 100'
cursor.execute(countstmt)
color = cursor.fetchall()
cursor.execute(selectstmt)
color2 = cursor.fetchall()
df = pd.DataFrame(columns = color, index = color)
the_matrix = np.array(color2)
shape = df.shape[0]
inputs = deltas(shape)
manage(inputs)
print (df)



