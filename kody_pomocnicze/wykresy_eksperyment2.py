import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime, timedelta

def celcjusz_to_kelwin(st):
    return st+273.15


def kelwin_to_celcjusz(st_k):
    return st_k-273.15


def srednia_tem_w_pokoju(mat):
    room = mat[1:-1, 1:-1]
    return np.mean(room)


U = np.load("U_doo1_" + str(3+1) + ".npy")
Ui = np.load("U_doo12_" + str(3+1) + ".npy")

h_x = 0.1

xs1 = np.arange(0,11,h_x)
ys1 = np.arange(0,5,h_x)
xs2 = np.arange(0,11,h_x)
ys2 = np.arange(5,7,h_x)
xs3 = np.arange(0,5,h_x)
ys3 = np.arange(7,11,h_x)
xs4 = np.arange(8,11,h_x)
ys4 = np.arange(7,11,h_x)

ys1_2 = list(ys1) + list(ys2)
y = ys1_2 + list(ys3)

# dla pierwszego układu
# Tworzenie układu podwykresów
fig, axs = plt.subplots(2, 3, figsize=(17.5, 10), sharey=True)
#fig.suptitle("Mapa ciepła dla pierwszego ułożenia grzejników", fontsize = 15)
fig.suptitle("")
# Tworzenie siatki współrzędnych
X, Y = np.meshgrid(xs1, y)

# Pierwszy wykres
im1 = axs[0, 0].pcolormesh(X, Y, U[:, :, 0], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 0].set_xlim([min(xs1), max(xs1)])
axs[0, 0].set_ylim([min(y), max(y)])
axs[0, 0].invert_yaxis()  # Odwrócenie osi y
axs[0, 0].set_title("t=0 (grzejniki ciągle działają)")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_xlabel("x")
fig.colorbar(im1, ax=axs[0, 0])

# Drugi wykres
im2 = axs[0, 1].pcolormesh(X, Y, U[:, :, 8], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 1].set_xlim([min(xs1), max(xs1)])
axs[0, 1].set_ylim([min(y), max(y)])
axs[0, 1].invert_yaxis()  # Odwrócenie osi y
axs[0, 1].set_title("t=2h (grzejniki ciągle działają)")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlabel("x")
fig.colorbar(im2, ax=axs[0, 1])

#trzeci wykres
im2 = axs[0, 2].pcolormesh(X, Y, U[:, :, 16], cmap='viridis')  # Bez globalnego vmin, vmax
axs[0, 2].set_xlim([min(xs1), max(xs1)])
axs[0, 2].set_ylim([min(y), max(y)])
axs[0, 2].invert_yaxis()  # Odwrócenie osi y
axs[0, 2].set_title("t=4h (grzejniki ciągle działają)")
axs[0, 2].set_ylabel("y")
axs[0, 2].set_xlabel("x")
fig.colorbar(im2, ax=axs[0, 2])

# czwarty wykres
im3 = axs[1, 0].pcolormesh(X, Y, Ui[:, :, 0], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 0].set_xlim([min(xs1), max(xs1)])
axs[1, 0].set_ylim([min(y), max(y)])
axs[1, 0].invert_yaxis()  # Odwrócenie osi y
axs[1, 0].set_title("t=0 (z wyłączeniem grzejników) ")
axs[1, 0].set_ylabel("y")
axs[1, 0].set_xlabel("x")
fig.colorbar(im3, ax=axs[1, 0])

# Piąty wykres
im4 = axs[1, 1].pcolormesh(X, Y, Ui[:, :, 8], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 1].set_xlim([min(xs1), max(xs1)])
axs[1, 1].set_ylim([min(y), max(y)])
axs[1, 1].invert_yaxis()  # Odwrócenie osi y
axs[1, 1].set_title(f"t=2h (z wyłączeniem grzejników)")
axs[1, 1].set_ylabel("y")
axs[1, 1].set_xlabel("x")
fig.colorbar(im4, ax=axs[1, 1])

# szósty wykres
im4 = axs[1, 2].pcolormesh(X, Y, Ui[:, :, 16], cmap='viridis')  # Bez globalnego vmin, vmax
axs[1, 2].set_xlim([min(xs1), max(xs1)])
axs[1, 2].set_ylim([min(y), max(y)])
axs[1, 2].invert_yaxis()  # Odwrócenie osi y
axs[1, 2].set_title(f"t=4h (z wyłączeniem grzejników)")
axs[1, 2].set_ylabel("y")
axs[1, 2].set_xlabel("x")
fig.colorbar(im4, ax=axs[1, 2])

plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace: odstęp w pionie, wspace: odstęp w poziomie

plt.show()



# dla 1 przypadku
with open("data3.json", "r") as sr_tem_po_1h:
    sr_tem_w_mieszkaniu = json.load(sr_tem_po_1h)

keys = []
#print(type(sr_tem_w_mieszkaniu))
for key in sr_tem_w_mieszkaniu:
    keys.append(key)  # Wyjście: name, age

sr_tem_pom1 = sr_tem_w_mieszkaniu[keys[0]]
sr_tem_pom2 = sr_tem_w_mieszkaniu[keys[1]]
sr_tem_pom3 = sr_tem_w_mieszkaniu[keys[2]]
sr_tem_pom4 = sr_tem_w_mieszkaniu[keys[3]]
calka_energii1 = sr_tem_w_mieszkaniu[keys[4]]

# dla 2 przypadku
with open("data4.json", "r") as sr_tem_po_1hi:
    sr_tem_w_mieszkaniui = json.load(sr_tem_po_1hi)

keyss = []
#print(type(sr_tem_w_mieszkaniu))
for key_i in sr_tem_w_mieszkaniui:
    keyss.append(key_i)  # Wyjście: name, age

sr_tem_pom1i = sr_tem_w_mieszkaniui[keyss[0]]
sr_tem_pom2i = sr_tem_w_mieszkaniui[keyss[1]]
sr_tem_pom3i = sr_tem_w_mieszkaniui[keyss[2]]
sr_tem_pom4i = sr_tem_w_mieszkaniui[keyss[3]]
calka_energii1i = sr_tem_w_mieszkaniui[keyss[4]]


# Godzina startowa
start_time = datetime.strptime("13:00:00", "%H:%M:%S")

# Lista danych w sekundach
seconds = [int(i_s/10) for i_s in range(0,len(sr_tem_pom1), 10*60*10)]

# Tworzenie listy z czasami
times = [start_time + timedelta(seconds=s) for s in seconds]

# Wyświetlenie wyników jako lista godzin (HH:MM:SS)
times_formatted = [t.strftime("%H:%M:%S") for t in times]
times_short = [t[:5] for t in times_formatted]
# Wyjście: ['08:00:00', '08:02:00', '08:05:00', '08:15:00']


# dla 1 układu
values1 = [kelwin_to_celcjusz(sr_tem_pom1[i_s]) for i_s in range(0,len(sr_tem_pom1), 10*60*10)]
values2 = [kelwin_to_celcjusz(sr_tem_pom2[i_s]) for i_s in range(0,len(sr_tem_pom1), 10*60*10)]
values3 = [kelwin_to_celcjusz(sr_tem_pom3[i_s]) for i_s in range(0,len(sr_tem_pom1), 10*60*10)]
values4 = [kelwin_to_celcjusz(sr_tem_pom4[i_s]) for i_s in range(0,len(sr_tem_pom1), 10*60*10)]
# dla 2 układu
values1i = [kelwin_to_celcjusz(sr_tem_pom1i[i_s]) for i_s in range(0,len(sr_tem_pom1i), 10*60*10)]
values2i = [kelwin_to_celcjusz(sr_tem_pom2i[i_s]) for i_s in range(0,len(sr_tem_pom1i), 10*60*10)]
values3i = [kelwin_to_celcjusz(sr_tem_pom3i[i_s]) for i_s in range(0,len(sr_tem_pom1i), 10*60*10)]
values4i = [kelwin_to_celcjusz(sr_tem_pom4i[i_s]) for i_s in range(0,len(sr_tem_pom1i), 10*60*10)]

plt.plot(times_short, values1, marker="o", color = "blue", label = "pokój dzienny (GW)")
plt.plot(times_short, values2, marker="o", color = "green", label = "korytarz (GW)")
plt.plot(times_short, values3, marker="o", color = "brown", label = "sypialnia (GW)")
plt.plot(times_short, values4, marker="o", color = "red", label = "łazienka (GW)")
plt.plot(times_short, values1i, marker="o", color = "lightblue", label = "pokój dzienny (GZ)")
plt.plot(times_short, values2i, marker="o", color = "purple", label = "korytarz (GZ)")
plt.plot(times_short, values3i, marker="o", color = "black", label = "sypialnia (GZ)")
plt.plot(times_short, values4i, marker="o", color = "orange", label = "łazienka (GZ)")
plt.legend(title="Legenda", loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Czas (HH:MM)")
plt.ylabel("Temperatura", fontsize=10)
#plt.title("Wykres temperatury w pokojach przy pierwszym układzie grzejników w zależności od czasu", fontsize=20)
plt.title("")
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()


# wspólny wykres oddanej energii

calka_suma = np.cumsum(calka_energii1)
values5 = [calka_suma[i_s] for i_s in range(0,len(sr_tem_pom1), 10*60*10)]

calka_sumai = np.cumsum(calka_energii1i)
values5i = [calka_sumai[i_s] for i_s in range(0,len(sr_tem_pom1i), 10*60*10)]

plt.plot(times_short, values5, label="włączone grzejniki", color="blue")
plt.plot(times_short, values5i, label="zgaszone grzejniki", color="red")
plt.xlabel("Czas (HH:MM)")
plt.ylabel("Oddana energia do układu")
plt.title("")
plt.legend(title="Legenda", loc="center left", bbox_to_anchor=(1, 0.5))
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()

