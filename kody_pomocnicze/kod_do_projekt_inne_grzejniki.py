import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


##########################
# jak robimy od pewnego momentu

# czas startowy w godzinach
t_0 = 3

# Wczytanie z pliku
U = np.load("U_doo_" + str(t_0) + ".npy")
u1_0 = np.load("u1_o" + str(t_0) + ".npy")
u2_0 = np.load("u2_o" + str(t_0) + ".npy")
u3_0 = np.load("u3_o" + str(t_0) + ".npy")
u4_0 = np.load("u4_o" + str(t_0) + ".npy")

with open("data2.json", "r") as sr_tem_po_1h_inne_grzejniki:
     sr_tem_w_mieszkaniu = json.load(sr_tem_po_1h_inne_grzejniki)

keys = []
print(type(sr_tem_w_mieszkaniu))
for key in sr_tem_w_mieszkaniu:
    keys.append(key)  # Wyjście: name, age


sr_tem_pom1 = sr_tem_w_mieszkaniu[keys[0]]
sr_tem_pom2 = sr_tem_w_mieszkaniu[keys[1]]
sr_tem_pom3 = sr_tem_w_mieszkaniu[keys[2]]
sr_tem_pom4 = sr_tem_w_mieszkaniu[keys[3]]
calka_energii = sr_tem_w_mieszkaniu[keys[4]]

###################################################


# funkcja dla czasu zero, warunek początkowy

def celcjusz_to_kelwin(st):
    return st+273.15


def kelwin_to_celcjusz(st_k):
    return st_k-273.15


tem_pocz = 13


def f_0(x, y):
    return celcjusz_to_kelwin(tem_pocz)


# parametry
T = 3600
h_t = 0.1
#t_0 = 0          ########################################### to wyf=gaszamy
h_x = 0.1
n=int(T/h_t)
alpha = 0.025

dane = pd.read_csv("C:\\Users\\klaud\\OneDrive\\Pulpit\\uczelnia\\matematyka\\sem1mgr\\modelowanie_deterministyczne\\temp_godz_projekt.csv")
temp_1 = list(dane.temperatura[:])


# Punkty danych
x_points = [0,3600]
y_points = temp_1[t_0+9:t_0+9+2]


def f_okna(time):
    return celcjusz_to_kelwin(np.interp(time, x_points, y_points))


# def x i y dla obszaru l1 i l2
xs1 = np.arange(0,11,h_x)
ys1 = np.arange(0,5,h_x)
#print(xs1)
#print(ys1)
xs2 = np.arange(0,11,h_x)
ys2 = np.arange(5,7,h_x)
#print(xs2)
#print(ys2)
xs3 = np.arange(0,5,h_x)
ys3 = np.arange(7,11,h_x)
#print(xs3)
#print(ys3)
xs4 = np.arange(8,11,h_x)
ys4 = np.arange(7,11,h_x)
#print(xs4)
#print(ys4)
# wymiary macierz
l1_x, l1_y = len(xs1), len(ys1)
#print(l1_x, l1_y)
l2_x, l2_y = len(xs2), len(ys2)
#print(l2_x, l2_y)
l3_x, l3_y = len(xs3), len(ys3)
#print(l3_x, l3_y)
l4_x, l4_y = len(xs4), len(ys4)
#print(l4_x, l4_y)

"""
###############################################
# def macierzy poczatkowych                                                     tylko kiedy zaczynamy robimy tę część
##############################
# uwaga trzeba dać x jako wsp. 2 a y jako pierwszą, by macierz miała sens względem osi
# bo wtedy x jest w poziomie, a y w pionie jak powinno być
##################################
u1_0 = np.zeros(shape=(l1_y, l1_x), dtype=None)
#print(u1_0.shape)
u2_0 = np.zeros(shape=(l2_y, l2_x), dtype=None)
#print(u2_0.shape)
u3_0 = np.zeros(shape=(l3_y, l3_x), dtype=None)
#print(u3_0.shape)
u4_0 = np.zeros(shape=(l4_y, l4_x), dtype=None)
#print(u4_0.shape)

for i in range(l1_x):
    u1_0[:,i] = [f_0(xs1[i], ys1[j]) for j in range(l1_y)]

for i in range(l2_x):
    u2_0[:,i] = [f_0(xs2[i], ys2[j]) for j in range(l2_y)]


for i in range(l3_x):
    u3_0[:,i] = [f_0(xs3[i], ys3[j]) for j in range(l3_y)]

for i in range(l4_x):
    u4_0[:,i] = [f_0(xs4[i], ys4[j]) for j in range(l4_y)]


# okna

# w 1 pokoju
# w kuchni
o_s11 = -int(1.51 / h_x)+1
o_e11 = -int(2.61/ h_x)
#print(o_s11, o_e11)
for o11 in range(o_e11, o_s11):
    #print(o11)
    u1_0[0, o11] = f_okna(0)
# w salonie
o_s12 = int(1.71 / h_x)
o_e12 = int(3.11 / h_x) + 1
for o12 in range(o_s12, o_e12):
    #print(o12)
    u1_0[o12, 0] = f_okna(0)
# w sypialni
o_s3 = int(1.41 / h_x)
o_e3 = int(2.51 / h_x) + 1
for o3 in range(o_s3, o_e3):
    #print(o3)
    u3_0[o3, 0] = f_okna(0)

#drzwi
# z 1 do 2 pokoju
i_s1 = int(5.21/h_x)
i_e1 = int(5.91/h_x) + 1
for j12 in range(i_s1, i_e1):
    #print(j12)
    a = u1_0[-2, j12]
    b = u2_0[1, j12]
    c = (a+b)/2
    u1_0[-1, j12], u2_0[0, j12] = c, c

# z 2 do 3 pokoju
i_s2 = int(3.71/h_x)
i_e2 = int(4.41/h_x) + 1
for j23 in range(i_s2, i_e2):
    #print(j23)
    a = u2_0[-2, j23]
    b = u3_0[1, j23]
    c = (a+b)/2
    u2_0[-1, j23], u3_0[0, j23] = c, c

# z 2 do 4 pokoju
i_s3 = -int(1.21/h_x)+1
i_e3 = -int(1.91/h_x)
for j24 in range(i_e3, i_s3):
    #print(j24)
    a = u2_0[-2, j24]
    b = u4_0[1, j24]
    c = (a+b)/2
    u2_0[-1, j24], u4_0[0, j24] = c, c

#print(l1_y+l2_x+l3_x-2)
# sklejanie podobszarów do jednego obszaru za pomocą macierzy
u_0 = np.zeros(shape=(l1_y+l2_y+l3_y, l1_x), dtype=None)
l_zer = [None]*(l1_x-l3_x-l4_x)
for i in range(l1_y+l2_y+l3_y):
    if i < l1_y:
        #print("dla pierwszego przypadku", i)
        u_0[i,] = u1_0[i,]
    elif l1_y <= i < l2_y+l1_y:
        #print("dla drugiego przypadku", i)
        u_0[i,] = u2_0[i-l1_y,]
    else:
        #print(i-l1_y-l2_y)
        u_0[i,] = list(u3_0[i-l1_y-l2_y,]) + l_zer + list(u4_0[i-l1_y-l2_y,])

############################# koniec części której wykonujemy tylko za 1 razem
"""


ys1_2 = list(ys1) + list(ys2)
y = ys1_2 + list(ys3)

"""
################# tylko za 1 razem ############
U = np.zeros((l1_y+l2_y+l3_y, l1_x, 24), None) # tworzenie całościowej macierzy rozwiązania dla wszystkich czasów
U[:, :, 0] = u_0 # dodanie warunku początkowego
####################################################################
"""

# grzejniki
P = 1200  # moc grzejnika
S = [celcjusz_to_kelwin(22), celcjusz_to_kelwin(20), celcjusz_to_kelwin(19), celcjusz_to_kelwin(24)]
ro_air = 1.205
c = 1005
R_i = 1.2*h_x*1 # dł x szer x wys_mieszkania
v = P/(ro_air*R_i*c)

def srednia_tem_w_pokoju(mat):
    room = mat[1:-1, 1:-1]
    return np.mean(room)


def f_grzejnik(nr_pok, sr_tem, i_x, i_y, calka):
    if nr_pok == 1:
    # 1 pokój- 3 grzejniki
        if (i_y==1 or i_y==48 ) and int(1.41/h_x)<=i_x<=int(2.51/h_x):
            if sr_tem<=S[0]:
                #print("grzejnik działa w 1 jeden z 2")
                return v
        elif i_x == l1_x-2 and int(0.31/h_x)<=i_y<=int(1.41/h_x):
            if sr_tem <= S[0]:
                #print("grzejnik działa w 1 ten drugi")
                return v

    elif nr_pok == 2:
    # 2 pokój
        if i_y==1 and int(8.51/h_x)<=i_x<=int(9.61/h_x):
            if sr_tem <= S[1]:
                #print("grzejnik działa w 2")
                return v

    elif nr_pok == 3:
    # 3 pokój
        if i_y==1 and int(1.41/h_x)<=i_x<=int(2.51/h_x):
            if sr_tem <= S[2]:
                #print("grzejnik działa w 3")
                return v

    elif nr_pok == 4:
    # 4 pokój
        if i_x==1 and int(0.31/h_x)<=i_y<=int(1.41/h_x):
            if sr_tem <= S[3]:
                #print("grzejnik działa w 4")
                return v
    return 0


# rozwiązanie L1
U_L1 = np.zeros((l1_y, l1_x, n+1), None)
U_L1[:, :, 0] = u1_0
#sr_tem_pom1 = []  ############################################################ przy kolejnych gasimy
# rozwiązanie L2
U_L2 = np.zeros((l2_y, l2_x, n+1), None)
U_L2[:, :, 0] = u2_0
#sr_tem_pom2 = []     ########################################################################################

# rozwiązanie L3
U_L3 = np.zeros((l3_y, l3_x, n+1), None)
U_L3[:, :, 0] = u3_0
#sr_tem_pom3 = []     ########################################################################################
# rozwiązanie L4
U_L4 = np.zeros((l4_y, l4_x, n+1), None)
U_L4[:, :, 0] = u4_0
#sr_tem_pom4 = []     ########################################################################################

#calka_energii = [0] ###########################################################################################

for k in range(n):
    calka_energii_cz = 0
    #print(k+1)
    # L1
    sr_tem_w_1 = srednia_tem_w_pokoju(U_L1[:,:,k])
    sr_tem_pom1.append(sr_tem_w_1)
    for i in range(l1_x):
        for j in range(l1_y):
            #print((i,j,k))
            if i + 1 > l1_x - 1 or j + 1 > l1_y - 1 or i - 1 < 0 or j - 1 < 0:
                U_L1[j, i, k + 1] = 0 # Dirichlet wszedzie, pozniej poprawiamy na Neumana
            else:
                v_f = f_grzejnik(1, sr_tem_w_1, i, j, calka_energii)
                calka_energii_cz += v_f
                U_L1[j, i, k+1] = ((U_L1[j, i, k] + alpha*h_t / (h_x ** 2) * (
                            U_L1[j, i + 1, k] + U_L1[j, i - 1, k] +
                            U_L1[j + 1, i, k] + U_L1[j - 1, i, k] - 4 * U_L1[j, i, k])) +
                                   h_t*v_f)
    U_L1[-1, :, k + 1] = U_L1[-2, :, k + 1] # dół
    U_L1[0, :, k + 1] = U_L1[1, :, k + 1]  # góra
    U_L1[:, -1, k + 1] = U_L1[:, -2, k + 1] # prawo
    U_L1[:, 0, k + 1] = U_L1[:, 1, k + 1] # lewo


    # L2
    sr_tem_w_2 = srednia_tem_w_pokoju(U_L2[:, :, k])
    sr_tem_pom2.append(sr_tem_w_2)
    for i2 in range(l2_x):
        for j2 in range(l2_y):
            if i2 + 1 > l2_x - 1 or j2 + 1 > l2_y - 1 or i2 - 1 < 0 or j2 - 1 < 0:
                U_L2[j2, i2, k + 1] = 0 # Dirichlet wszedzie, pozniej poprawiamy na Neumana
            else:
                v_f = f_grzejnik(2,sr_tem_w_2, i2, j2, calka_energii)
                calka_energii_cz += v_f
                U_L2[j2, i2, k+1] = (U_L2[j2, i2, k] + alpha*h_t / (h_x ** 2) * (
                            U_L2[j2, i2 + 1, k] + U_L2[j2, i2 - 1, k] +
                            U_L2[j2 + 1, i2, k] + U_L2[j2 - 1, i2, k] - 4 * U_L2[j2, i2, k]) +
                                     h_t*v_f)
    U_L2[0, :, k + 1] = U_L2[1, :, k + 1]  # góra
    U_L2[-1, :, k + 1] = U_L2[-2, :, k + 1]  # dół
    U_L2[:, -1, k + 1] = U_L2[:, -2, k + 1] # prawo
    U_L2[:, 0, k + 1] = U_L2[:, 1, k + 1] # lewo

    # L3
    sr_tem_w_3 = srednia_tem_w_pokoju(U_L3[:, :, k])
    sr_tem_pom3.append(sr_tem_w_3)
    for i3 in range(l3_x):
        for j3 in range(l3_y):
            if i3 + 1 > l3_x - 1 or j3 + 1 > l3_y - 1 or i3 - 1 < 0 or j3 - 1 < 0:
                U_L3[j3, i3, k + 1] = 0 # Dirichlet wszedzie, pozniej poprawiamy na Neumana
            else:
                v_f = f_grzejnik(3, sr_tem_w_3, i3, j3, calka_energii)
                calka_energii_cz += v_f
                U_L3[j3, i3, k+1] = ((U_L3[j3, i3, k] + alpha*h_t / (h_x ** 2) * (
                            U_L3[j3, i3 + 1, k] + U_L3[j3, i3 - 1, k] +
                            U_L3[j3 + 1, i3, k] + U_L3[j3 - 1, i3, k] - 4 * U_L3[j3, i3, k])) +
                                     h_t*v_f)

    U_L3[0, :, k + 1] = U_L3[1, :, k + 1]  # góra
    U_L3[-1, :, k + 1] = U_L3[-2, :, k + 1]  # dół
    U_L3[:, -1, k + 1] = U_L3[:, -2, k + 1]  # prawo
    U_L3[:, 0, k + 1] = U_L3[:, 1, k + 1]  # lewo


    # L4
    sr_tem_w_4 = srednia_tem_w_pokoju(U_L4[:, :, k])
    sr_tem_pom4.append(sr_tem_w_4)
    for i4 in range(l4_x):
        for j4 in range(l4_y):
            if i4 + 1 > l4_x - 1 or j4 + 1 > l4_y - 1 or i4 - 1 < 0 or j4 - 1 < 0:
                U_L4[j4, i4, k + 1] = 0  # Dirichlet wszedzie, pozniej poprawiamy na Neumana
            else:
                v_f = f_grzejnik(4, sr_tem_w_4, i4, j4, calka_energii)
                calka_energii_cz += v_f
                U_L4[j4, i4, k+1] = ((U_L4[j4, i4, k] + alpha*h_t / (h_x ** 2) * (
                            U_L4[j4, i4 + 1, k] + U_L4[j4, i4 - 1, k] +
                            U_L4[j4 + 1, i4, k] + U_L4[j4 - 1, i4, k] - 4 * U_L4[j4, i4, k])) +
                                     h_t*v_f)
    #print("calka_czesciowa"+str(calka_energii))
    U_L4[0, :, k + 1] = U_L4[1, :, k + 1]  # góra
    U_L4[-1, :, k + 1] = U_L4[-2, :, k + 1]  # dół
    U_L4[:, -1, k + 1] = U_L4[:, -2, k + 1]  # prawo
    U_L4[:, 0, k + 1] = U_L4[:, 1, k + 1]  # lewo

    # okna
    # w 1 pokoju
    # w kuchni
    o_s11 = -int(1.51 / h_x) + 1
    o_e11 = -int(2.61 / h_x)
    # print(o_s11, o_e11)
    for o11 in range(o_e11, o_s11):
        U_L1[0, o11, k+1] = f_okna(k+1)
    # duże w salonie
    o_s12 = int(1.71 / h_x)
    o_e12 = int(3.11 / h_x) + 1
    for o12 in range(o_s12, o_e12):
        U_L1[o12, 0, k+1] = f_okna(k+1)
    # w sypialni
    o_s3 = int(1.41 / h_x)
    o_e3 = int(2.51 / h_x) + 1
    for o3 in range(o_s3, o_e3):
        U_L3[o3, 0, k + 1] = f_okna(k + 1)

    #drzwi
    # z 1 do 2 pokoju
    i_s1 = int(5.21 / h_x)
    i_e1 = int(5.91 / h_x) + 1
    for j12 in range(i_s1, i_e1):
        a = U_L1[-2, j12, k+1]
        b = U_L2[1, j12, k+1]
        c = (a+b)/2
        U_L1[-1, j12, k+1], U_L2[0, j12, k+1] = c, c

    # z 2 do 3 pokoju
    i_s2 = int(3.71 / h_x)
    i_e2 = int(4.41 / h_x) + 1
    for j23 in range(i_s2, i_e2):
        a = U_L2[-2, j23, k+1]
        b = U_L3[1, j23, k+1]
        c = (a+b)/2
        U_L2[-1, j23, k+1], U_L3[0, j23, k+1] = c, c

    # z 2 do 4 pokoju
    i_s3 = -int(1.21 / h_x) + 1
    i_e3 = -int(1.91 / h_x)
    for j24 in range(i_e3, i_s3):
        a = U_L2[-2, j24, k+1]
        b = U_L4[1, j24, k+1]
        c = (a+b)/2
        U_L2[-1, j24, k+1], U_L4[0, j24, k+1] = c, c

    calka_energii.append(calka_energii_cz*h_x*h_t*h_x)


# szukanie, kiedy grzejniki przestają grzać

def find_first_greater_unsorted(lst, target):
    for i, val in enumerate(lst):
        if val > target:
            return i, val
    return None  # Jeśli nie ma elementu większego niż target

print("czy kiedyś osiągnięto pożądaną temperaturę?")
print(find_first_greater_unsorted(sr_tem_pom1, S[0]))
print(find_first_greater_unsorted(sr_tem_pom2, S[1]))
print(find_first_greater_unsorted(sr_tem_pom3, S[2]))
print(find_first_greater_unsorted(sr_tem_pom4, S[3]))

print("średnia tem dla ostatniego pomiaru")
print(sr_tem_pom1[-1])
print(sr_tem_pom2[-1])
print(sr_tem_pom3[-1])
print(sr_tem_pom4[-1])

sr_tem_w_mieszkaniu={1:sr_tem_pom1, 2:sr_tem_pom2, 3:sr_tem_pom3, 4:sr_tem_pom4, "calka":calka_energii}
# dodawanie dwóch obszarów rozwiązanych do jednej macierzy
ii = 0
for index in [900, 1800, 2700, 3600]:
    ii+=1
    u_k = np.zeros(shape=(l1_y+l2_y+l3_y, l1_x), dtype=None)
    l_zer = [None]*(l1_x-l3_x-l4_x)
    for i in range(l1_y+l2_y+l3_y):
        if i < l1_y:
            u_k[i,] = U_L1[i,:,index]
        elif l1_y <= i < l2_y + l1_y:
            u_k[i,] = U_L2[i - l1_y, :, index]
        else:
            u_k[i, ] = list(U_L3[i-l1_y-l2_y, :, index]) + l_zer + list(U_L4[i-l1_y-l2_y, :, index])

    U[:, :, 4*t_0+ii] = u_k

# Zapis do pliku
np.save("U_doo_"+ str(t_0+1)+".npy", U)
np.save("u1_o"+ str(t_0+1), U_L1[:,:,-1])
np.save("u2_o"+ str(t_0+1), U_L2[:,:,-1])
np.save("u3_o"+ str(t_0+1), U_L3[:,:,-1])
np.save("u4_o"+ str(t_0+1), U_L4[:,:,-1])


with open("data2.json", "w") as sr_tem_po_1h_inne_grzejniki:
    json.dump(sr_tem_w_mieszkaniu, sr_tem_po_1h_inne_grzejniki)


# Tworzenie układu podwykresów
fig, axs = plt.subplots(2, 2, figsize=(17.5, 10), sharey=True)
fig.suptitle("Mapa ciepła przy zmienionym ułożeniu grzejników")

# Tworzenie siatki współrzędnych
X, Y = np.meshgrid(xs1, y)

# Pierwszy wykres
im1 = axs[0, 0].pcolormesh(X, Y, U[:, :, 4*t_0], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 0].set_xlim([min(xs1), max(xs1)])
axs[0, 0].set_ylim([min(y), max(y)])
axs[0, 0].invert_yaxis()  # Odwrócenie osi y
axs[0, 0].set_title("t="+str(t_0)+"h")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_xlabel("x")
fig.colorbar(im1, ax=axs[0, 0])

# Drugi wykres
im2 = axs[0, 1].pcolormesh(X, Y, U[:, :, 4*t_0+1], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 1].set_xlim([min(xs1), max(xs1)])
axs[0, 1].set_ylim([min(y), max(y)])
axs[0, 1].invert_yaxis()  # Odwrócenie osi y
axs[0, 1].set_title("t="+str(t_0)+"h15min")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlabel("x")
fig.colorbar(im2, ax=axs[0, 1])

# Trzeci wykres
im3 = axs[1, 0].pcolormesh(X, Y, U[:, :, 4*t_0+2], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 0].set_xlim([min(xs1), max(xs1)])
axs[1, 0].set_ylim([min(y), max(y)])
axs[1, 0].invert_yaxis()  # Odwrócenie osi y
axs[1, 0].set_title("t="+str(t_0)+"h30min ")
axs[1, 0].set_ylabel("y")
axs[1, 0].set_xlabel("x")
fig.colorbar(im3, ax=axs[1, 0])

# Czwarty wykres
im4 = axs[1, 1].pcolormesh(X, Y, U[:, :, 4*t_0+3], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 1].set_xlim([min(xs1), max(xs1)])
axs[1, 1].set_ylim([min(y), max(y)])
axs[1, 1].invert_yaxis()  # Odwrócenie osi y
axs[1, 1].set_title(f"t="+str(t_0)+"h45min")
axs[1, 1].set_ylabel("y")
axs[1, 1].set_xlabel("x")
fig.colorbar(im4, ax=axs[1, 1])

plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace: odstęp w pionie, wspace: odstęp w poziomie

plt.show()

"""

# Pierwszy wykres
im1 = axs[0, 0].pcolormesh(X, Y, U[:, :, 4*t_0], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 0].set_xlim([min(xs1), max(xs1)])
axs[0, 0].set_ylim([min(y), max(y)])
axs[0, 0].invert_yaxis()  # Odwrócenie osi y
axs[0, 0].set_title("t=0")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_xlabel("x")
fig.colorbar(im1, ax=axs[0, 0])

# Drugi wykres
im2 = axs[0, 1].pcolormesh(X, Y, U[:, :, 4*t_0+1], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 1].set_xlim([min(xs1), max(xs1)])
axs[0, 1].set_ylim([min(y), max(y)])
axs[0, 1].invert_yaxis()  # Odwrócenie osi y
axs[0, 1].set_title("t=15min")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlabel("x")
fig.colorbar(im2, ax=axs[0, 1])

# Trzeci wykres
im3 = axs[1, 0].pcolormesh(X, Y, U[:, :, 4*t_0+2], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 0].set_xlim([min(xs1), max(xs1)])
axs[1, 0].set_ylim([min(y), max(y)])
axs[1, 0].invert_yaxis()  # Odwrócenie osi y
axs[1, 0].set_title("t=30hmin ")
axs[1, 0].set_ylabel("y")
axs[1, 0].set_xlabel("x")
fig.colorbar(im3, ax=axs[1, 0])

# Czwarty wykres
im4 = axs[1, 1].pcolormesh(X, Y, U[:, :, 4*t_0+3], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 1].set_xlim([min(xs1), max(xs1)])
axs[1, 1].set_ylim([min(y), max(y)])
axs[1, 1].invert_yaxis()  # Odwrócenie osi y
axs[1, 1].set_title(f"t=45min")
axs[1, 1].set_ylabel("y")
axs[1, 1].set_xlabel("x")
fig.colorbar(im4, ax=axs[1, 1])

plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace: odstęp w pionie, wspace: odstęp w poziomie

plt.show()
"""

