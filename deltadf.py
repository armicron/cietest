import numpy as np
import mysql.connector
import pandas as pd
import theano
import theano.tensor as T
import unittest


def deltae_theano(lab_color_vector, lab_color_matrix):
    lab_vector = T.dvector("lab_vector")
    lab_matrix = T.dmatrix("lab_matrix")
    f = theano.function([lab_vector, lab_matrix],
                        (lab_vector[0] + lab_matrix[:,0]) / 2.0)
    avg_Lp = f(lab_color_vector, lab_color_matrix)
    f = theano.function([lab_vector], T.sqrt(T.sum(lab_vector[1:]**2)))
    C1 = f(lab_color_vector)
    f = theano.function([lab_matrix], T.sqrt(T.sum(lab_matrix[:, 1:]**2, axis=1)))
    C2 = f(lab_color_matrix)
    C1t = T.dscalar("C1t")
    C2t = T.dvector("C2t")
    f = theano.function([C1t,C2t], (C1t+C2t) / 2.0)
    avg_C1_C2 = f(C1, C2)
    avg_C1_C2t = T.dvector("avg_C1_C2t")
    f = theano.function([avg_C1_C2t],
                        0.5 * (1 - T.sqrt(avg_C1_C2t**7.0 / (avg_C1_C2t**7 + 25**7))))
    G = f(avg_C1_C2)
    Gt = T.dvector("Gt")
    f = theano.function([Gt,lab_vector], (1.0 + Gt) * lab_vector[1])
    a1p = f(G, lab_color_vector)
    f = theano.function([Gt, lab_matrix], (1.0 + Gt) * lab_matrix[:, 1])
    a2p = f(G, lab_color_matrix)
    a1pt = T.dvector("a1pt")
    f = theano.function([a1pt, lab_vector], T.sqrt(a1pt**2 + lab_vector[2]**2))
    C1p = f(a1p, lab_color_vector)
    a2pt = T.dvector("a2pt")
    f = theano.function([a2pt, lab_matrix], T.sqrt(a2pt**2 + lab_matrix[:, 2]**2))
    C2p = f(a2p, lab_color_matrix)
    C1pt = T.dvector("C1pt")
    C2pt = T.dvector("C2pt")
    f = theano.function([C1pt, C2pt], (C1pt + C2pt) / 2.0)
    avg_C1p_C2p = f(C1p, C2p)
    f = theano.function([lab_vector, a1pt], T.rad2deg(T.arctan2(lab_vector[2], a1pt)))
    h1p = f(lab_color_vector, a1p)
    h1pt = T.dvector("h1pt")
    f = theano.function([h1pt], h1pt + (h1pt < 0) * 360)
    h1p = f(h1p)
    f = theano.function([lab_matrix, a2pt], T.rad2deg(T.arctan2(lab_matrix[:, 2], a2pt)))
    h2p = f(lab_color_matrix, a2p)
    h2pt = T.dvector("h2pt")
    f = theano.function([h2pt], h2pt + (h2pt < 0) * 360)
    h2p = f(h2p)
    f = theano.function([h1pt,h2pt], (((T.abs_(h1pt - h2pt) > 180) * 360) + h1pt
                                      + h2pt) / 2.0)
    avg_Hp = f(h1p, h2p)
    avg_Hpt = T.dvector("avg_Hpt")
    f = theano.function([avg_Hpt], 1 - 0.17 * T.cos(T.deg2rad(avg_Hpt-30)) +
                        0.24 * T.cos(T.deg2rad(2 * avg_Hpt)) +
                        0.32 * T.cos(T.deg2rad(3 * avg_Hpt + 6)) -
                        0.2 * T.cos(T.deg2rad(4 * avg_Hpt - 63)))
    TT = f(avg_Hp)
    f = theano.function([h1pt, h2pt], h2pt - h1pt)
    diff_h2p_h1p = f(h1p, h2p)
    diff_h2p_h1pt = T.dvector("diff_h2p_h1pt")
    f = theano.function([diff_h2p_h1pt], diff_h2p_h1pt +
                        (T.abs_(diff_h2p_h1pt) > 180) * 360)
    delta_hp = f(diff_h2p_h1p)
    delta_hpt = T.dvector("delta_hpt")
    f = theano.function([h1pt, h2pt, delta_hpt], delta_hpt - (h2pt > h1pt) * 720)
    delta_hp = f(h1p, h2p, delta_hp)
    f = theano.function([lab_matrix, lab_vector], lab_matrix[:, 0] - lab_vector[0])
    delta_Lp = f(lab_color_matrix, lab_color_vector)
    f = theano.function([C1pt, C2pt], C2pt - C1pt)
    delta_Cp = f(C1p, C2p)
    f = theano.function([C1pt, C2pt, delta_hpt], 2 * T.sqrt(C2pt * C1pt) *
                        T.sin(T.deg2rad(delta_hpt) / 2.0))
    delta_Hp = f(C1p, C2p, delta_hp)
    avg_Lpt = T.dvector("avg_Lpt")
    f = theano.function([avg_Lpt], 1 + ((0.015 * (avg_Lpt - 50)**2)) /
                        T.sqrt(20 + (avg_Lpt - 50)**2))
    S_L = f(avg_Lp)
    avg_C1p_C2pt = T.dvector("avg_C1p_C2pt")
    f = theano.function([avg_C1p_C2pt], 1 + 0.045 * avg_C1p_C2pt)
    S_C = f(avg_C1p_C2p)
    TTt = T.dvector("TTt")
    f = theano.function([avg_C1p_C2pt, TTt], 1 + 0.015 * avg_C1p_C2pt * TTt)
    S_H = f(avg_C1p_C2p, TT)
    f = theano.function([avg_Hpt], 30 * T.exp(-((avg_Hpt - 275) / 25)**2))
    delta_ro = f(avg_Hp)
    f = theano.function([avg_C1p_C2pt], T.sqrt(avg_C1p_C2pt**7/
                                               (avg_C1p_C2pt**7 + 25**7)))
    R_C = f(avg_C1p_C2p)
    delta_rot = T.dvector("delta_rot")
    R_Ct = T.dvector("R_Ct")
    f = theano.function([delta_rot, R_Ct], -2 * R_Ct * T.sin(2 * T.deg2rad(delta_rot)))
    R_T = f(delta_ro, R_C)
    delta_Lpt = T.dvector("delta_Lpt")
    S_Lt = T.dvector("S_Lt")
    delta_Cpt = T.dvector("delta_Cpt")
    S_Ct = T.dvector("S_Ct")
    delta_Hpt = T.dvector("delta_Hpt")
    S_Ht = T.dvector("S_Ht")
    R_Tt = T.dvector("R_Tt")
    f = theano.function([delta_Lpt, S_Lt, delta_Cpt, S_Ct, delta_Hpt, S_Ht, R_Tt],
                        T.sqrt((delta_Lpt / S_Lt)**2 + (delta_Cpt / S_Ct)**2 +
                        (delta_Hpt / S_Ht)**2 +
                               R_Tt * (delta_Cpt / S_Ct) * (delta_Hpt/S_Ht)))
    result = f(delta_Lp, S_L, delta_Cp, S_C, delta_Hp, S_H, R_T)
    return result


def deltae_cie_2000(lab_color_vector, lab_color_matrix):
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
        np.power(delta_Lp / (S_L), 2) +
        np.power(delta_Cp / (S_C), 2) +
        np.power(delta_Hp / (S_H), 2) +
        R_T * (delta_Cp / (S_C)) * (delta_Hp / (S_H)))

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
        delta_e = deltae_theano(lab_color_vector, lab_color_matrix)
        df.ix[lab1idx] = delta_e

cnx = mysql.connector.connect(user='user', password='password', host='localhost', database='db_name')
cursor = cnx.cursor()
selectstmt = 'SELECT CIE_L, CIE_a, CIE_b FROM `db_name`.`B2Colors` LIMIT 100'
countstmt = 'SELECT colorID, CIE_L, CIE_a, CIE_b FROM `db_name`.`B2Colors` LIMIT 100'
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



class TestDelta(unittest.TestCase):

    def test_delta_cie_2000(self):
        lab_color_vector = np.array([69.6225, -54.1511, 67.9207])
        lab_color_matrix = np.array([[69.6225, -54.1511, 67.9207],
                                  [58.1785, -52.2517, 56.7720000],
                                  [87.0254, -32.1576, 66.4509]])
        test_value = np.array([0, 10.02693299, 14.6524302])
        result = deltae_cie_2000(lab_color_vector, lab_color_matrix)
        eps = np.full_like(result, 0.000001)
        self.assertTrue(np.all(abs(result - test_value) < eps))

    def test_equality(self):
        lab_color_vector = np.array([69.6225, -54.1511, 67.9207])
        lab_color_matrix = np.array([[69.6225, -54.1511, 67.9207],
                                  [58.1785, -52.2517, 56.7720000],
                                  [87.0254, -32.1576, 66.4509]])
        res_theano = deltae_theano(lab_color_vector, lab_color_matrix)
        print(res_theano)
        res_numpy = deltae_cie_2000(lab_color_vector, lab_color_matrix)
        print(res_numpy)
        eps = np.full_like(res_numpy, 0.000001)
        self.assertTrue(np.all(abs(res_theano - res_numpy) < eps))
