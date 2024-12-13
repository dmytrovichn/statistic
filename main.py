import random
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import math

from collections import Counter


plt.rcParams["figure.figsize"] = (15, 15)


def interval_emp_func(inter_ser, w_i, vybirka, sample_text):
    n = len(vybirka)
    r = 0
    res = 0

    while (True):
        if (2 ** r) < n and n <= (2 ** (r + 1)):
            res = r + 1
            break
        else:
            r += 1
            continue

    result_text = '\n\n---Емпірична функція інтервального розподілу---\n'

    s = [0]
    a = 0
    for i in range(len(w_i)):
        a += w_i[i]
        s.append(round(a, 2))

    result_text += f'0, x <= {inter_ser[0][0]}\n'
    for i in range(len(inter_ser)):
        result_text += f'{round (w_i[i] / len(vybirka), 2)} (x-{inter_ser[i][0]}) + {round(s[i] / len(vybirka), 2)}, {inter_ser[i][0]}<×<={inter_ser[i][1]};\n'
    result_text += f'1, x >= {inter_ser[-1][1]}\n'
    sample_text.insert(tk.END, result_text)

    counts, bins = np.histogram(vybirka, bins=res)
    intervals = []
    for i in range(len(bins)):
        bins[i] = round(bins[i], 2)

    a = 0
    b = 1
    for i in range(len(bins) - 1):
        intervals.append([bins[a], bins[b], round(counts[a] / len(vybirka), 2)])
        a += 1
        b += 1

    s = [0]
    a = 0
    for i in range(len(intervals)):
        a += intervals[i][2]
        s.append(round(a, 2))

    y = []
    x = [intervals[0][0]]
    for i in range(len(intervals)):
        x.append(intervals[i][1])
    for i in range(len(intervals)):
        y.append(((intervals[i][2] / len(vybirka)) * (x[i] - intervals[i][0])) + s[i])

    y.append(1)
    plt.plot(x, y)
    plt.show()


def harakter_inter(inter_ser, w_i, vyb, sample_text):
    result_text = '\n\n---Числові характеристики інтервального розподілу---\n'
    n = len(vyb)
    nmo, nmo_, nmo_1 = 0, 0, 0
    ind = 0
    for i in range(len(w_i)):
        if w_i[i] > nmo:
            nmo = w_i[i]
            nmo_ = w_i[i-1]
            nmo_1 = w_i[i+1]
            ind = i
    res1 = (nmo-nmo_)/((nmo-nmo_)+(nmo-nmo_1))
    res2 = inter_ser[ind][1]-inter_ser[ind][0]
    Mo = round(inter_ser[ind][0] + res1*res2, 2)
    result_text += f'Мода: {Mo}\n'

    res3 = 0
    for i in range(len(w_i)):
        res3 += ((inter_ser[i][0] + inter_ser[i][1]) / 2) * w_i[i]
    X_ = round(res3 / n, 0)
    result_text += f'Вибіркове середнє: {X_}\n'

    mm = 0
    res = len(w_i) % 2
    if(res != 0):
        index = math.trunc(len(w_i)/2)

    else:
        index = math.trunc(len(w_i)/2)
    for j in range(0,index):
        mm += w_i[j]
    res1_ = (inter_ser[index][1]-inter_ser[index][0])/w_i[index]
    res2_ = (n/2)-mm
    Me = round(inter_ser[index][0] + res1_*res2_,2)
    result_text += f'Медіана: {Me}\n'

    res4 = 0
    for i in range(len(w_i)):
        res4 += pow(((inter_ser[i][0]+inter_ser[i][1])/2)-X_,2) * w_i[i]
    dev = round(res4,0)
    result_text += f'Девіація: {dev}\n'

    s_2 = round(dev/(n-1),2)
    result_text += f'Варіанса: {s_2}\n'

    s = round(pow(s_2,1/2),2)
    result_text += f'Стандарт: {s}\n'

    v = round(s/X_,2)
    result_text += f'Варіація: {v}\n'

    D_ = round(dev/n,2)
    result_text += f'Вибіркова дисперсія: {D_}\n'

    Q_ = round(pow(D_,1/2),2)
    result_text += f'Вибіркове середнє квадратичне відхилення: {Q_}\n'

    m_1 = round(X_, 3)
    result_text += f'\n1-й початковий момент: {m_1}\n'

    m_2 = round(D_, 3)
    result_text += f'2-й центральний момент: {m_2}\n'

    res5 = 0
    for i in range(len(w_i)):
        res5 += pow(((inter_ser[i][0] + inter_ser[i][1]) / 2) - X_, 3) * w_i[i]
    res = round(res5, 0)
    m_3 = round(res/n, 2)
    result_text += f'3-й центральний момент: {m_3}\n'

    res6 = 0
    for i in range(len(w_i)):
        res6 += pow(((inter_ser[i][0] + inter_ser[i][1]) / 2) - X_, 4) * w_i[i]
    res7 = round(res6, 0)
    m_4 = round(res7 / n, 2)
    result_text += f'4-й центральний момент: {m_4}\n\n'

    A_s = m_3 / pow(m_2, 3/2)
    asemetria = ''
    if A_s > 0:
        asemetria = 'cтатистичний матеріал скошений вправо'
    if A_s < 0:
        asemetria = 'cтатистичний матеріал скошений вліво'
    if A_s == 0:
        asemetria = 'cтатистичний матеріал симетричний відносно середини проміжку'
    result_text += f'Асиметрія: {round(A_s, 3)}, {asemetria}\n'

    E_k = (m_4 / pow(m_2, 2)) - 3
    eks = ''
    if E_k > 0:
        eks = 'cтатистичний матеріал – високовершинний'
    if E_k < 0:
        eks = 'cтатистичний матеріал – низьковершинний'
    if E_k == 0:
        eks = 'cтатистичний матеріал – нормальновершинний'
    result_text += f'Ексцес: {round(E_k, 3)}, {eks}\n'
    sample_text.insert(tk.END, result_text)


def interval_distr(xi, vybirka, sample_text):
    result_text = '\n\n---Інтервальний розподіл---\n'
    vybirka = sorted(vybirka)
    n = len(vybirka)
    r = 0
    res = 0

    while True:
        if (2 ** r) < n and n <= (2 ** (r + 1)):
            res = r + 1
            break
        else:
            r += 1
            continue

    a = vybirka[0]
    b = vybirka[-1]
    interval = (b - a) / res
    inter_ser = []
    inter2 = interval
    inter3 = interval+interval

    for i in range(res):
        if i == 0:
            inter_ser.append([xi[i], xi[i]+interval])
        if i > 0:
            inter_ser.append([round(xi[0]+interval, 3), round(xi[0]+inter3, 3)])
            interval += inter2
            inter3 += inter2

    repet = Counter(vybirka)
    a = []
    for item in repet:
        a.append(repet[item])

    w_i = []
    ad = []
    for k in range(len(inter_ser)):
        summa = 0
        for i in repet:
            if i > inter_ser[k][0] and i <= inter_ser[k][1]:
                ad.append(repet.get(i))
            if k == 0:
                if i == inter_ser[k][0]:
                    ad.append(repet.get(i))

            summa = sum((int(ad[i]) for i in range((len(ad)))))
        w_i.append(summa)
        ad = []

    result_text += '(h_i-1; h_i) | w_i\n'
    for s, w in zip(inter_ser, w_i):
        result_text += f'({round(s[0], 3)}; {round(s[1], 2)}) | {w}\n'

    plt.title('Гістограма інтервального розподілу')
    data = {'vyb': vybirka}
    interv = inter2 + 0.0001
    plt.hist(data['vyb'], bins=np.arange(xi[0], xi[-1]+interv, interv), edgecolor='black', color='green')
    plt.show()

    sample_text.insert(tk.END, result_text)
    return inter_ser, w_i


def harakter_dyskr(xi, ni, vyb, sample_text):
    result_text = '\n\n---Числові характеристики дискретного розподілу---\n'
    max_n = 0
    ind_n = 0
    for i in range(len(ni)):
        if max_n < ni[i]:
            max_n = ni[i]
            ind_n = i

    moda = round(xi[ind_n], 4)
    result_text += f'Мода: {moda}\n'

    sum_x = sum(xi[i]*ni[i] for i in range(len(xi)))
    x_avg = round(sum_x/len(vyb), 4)
    result_text += f'Вибіркове середнє: {x_avg}\n'

    len_vyb = len(vyb)
    sum1 = 0
    index1 = 0
    for i in range(len(ni)):
        temp2 = len_vyb / 2
        sum1 += ni[i]
        if temp2 < sum1:
            index1 = i
            break
    median = round(xi[index1], 4)
    result_text += f'Медіана: {median}\n'

    p = round(xi[len(xi)-1] - xi[0], 4)
    result_text += f'Розмах вибірки: {p}\n'

    dev = 0
    for i in range(len(xi)):
        dev += ni[i]*pow((xi[i]-x_avg), 2)
    dev = round(dev, 4)
    result_text += f'Девіація: {dev}\n'

    s_2 = dev/(len_vyb-1)
    result_text += f'Варіанса: {round(s_2, 4)}\n'

    s = pow(s_2, 1/2)
    result_text += f'Стандарт: {round(s, 4)}\n'

    v = s/x_avg
    result_text += f'Варіація: {round(v, 4)}\n'

    despersion= dev/len_vyb
    result_text += f'Вибіркова дисперія: {round(despersion, 4)}\n'

    avg_q = pow(despersion, 1/2)
    result_text += f'Вибіркове середнє квадратичне відхилення: {round(avg_q,4)}\n\n'

    if len_vyb % 4 == 0:
        qw4 = []
        for i in range(0, 5):
            fq4 = len_vyb / 4
            sumf41 = 0
            sumf42 = 0
            fq4 = i * fq4
            for j in range(len(ni)):
                sumf42 += ni[j]
                if sumf41<fq4 and fq4<=sumf42:
                    inq = j
                    qw4.append(xi[inq])
                sumf41 += ni[j]
        for i in range(3):
            if i == 2:
                result_text += f'Інтерквантильна широта: {qw4[i] - qw4[0]}\n'
                print()
    else:
        result_text += 'Квантиль знайти неможливо!\n'

    if len_vyb%8 ==0:
        qw8 = []
        for i in range(0, 9):
            fq8 = len_vyb / 8
            sumf81 = 0
            sumf82 = 0
            fq8 = i * fq8
            for j in range(len(ni)):
                sumf82 += ni[j]
                if sumf81 < fq8 and fq8 <= sumf82:
                    inq2 = j
                    qw8.append(xi[inq2])
                sumf81 += ni[j]
        for i in range(7):
            if i == 6:
                result_text += f'Інтероктильнатильна широта: {qw8[i] - qw8[0]}\n'
    else:
        result_text += 'Октиль знайти неможливо!\n'

    if len_vyb%10 ==0:
        qw10 = []
        for i in range(0, 11):
            fq10 = len_vyb / 10
            sumf101 = 0
            sumf102 = 0
            fq10 = i * fq10
            for j in range(len(ni)):
                sumf102 += ni[j]
                if sumf101 < fq10 and fq10 <= sumf102:
                    inq2 = j
                    qw10.append(xi[inq2])
                sumf101 += ni[j]
        for i in range(9):
            if i == 8:
                result_text += f'Інтердецильнальна широта: {qw10[8] - qw10[0]}\n'
    else:
        result_text += 'Дециль знайти неможливо!\n'

    if len_vyb % 100 == 0:
        qw100 = []
        for i in range(0, 101):
            fq100 = len_vyb / 100
            sumf1001 = 0
            sumf1002 = 0
            fq100 = i * fq100
            for j in range(len(ni)):
                sumf1002 += ni[j]
                if sumf1001 < fq100 and fq100 <= sumf1002:
                    inq2 = j
                    qw100.append(xi[inq2])
                sumf1001 += ni[j]
        for i in range(99):
            if i == 98:
                result_text += f'Інтерцентильна широта: {qw100[i] - qw100[0]}\n'
    else:
        result_text += 'Центиль знайти неможливо!\n'

    if len_vyb % 1000 ==0:
        qw1000 = []
        for i in range(0, 1001):
            fq100 = len_vyb / 1000
            sumf10001 = 0
            sumf10002 = 0
            fq100 = i * fq100
            for j in range(len(ni)):
                sumf10002 += ni[j]
                if sumf10001 < fq100 and fq100 <= sumf10002:
                    inq2 = j
                    qw1000.append(xi[inq2])
                sumf10001 += ni[j]
        for i in range(999):
            if i == 998:
                result_text += f'Інтермілільна широта: {qw1000[i] - qw1000[0]}\n'
    else:
        result_text += 'Міліль знайти неможливо!\n'

    result_text += f'\n1-й початковий момент: {round(x_avg,4)}\n'
    m_2 = ((len_vyb-1)/len_vyb)*s_2
    result_text += f'2-й центральний момент: {round(m_2, 4)}\n'
    temp3 = 0
    for i in range(len(xi)):
        temp3 += ni[i]*pow(xi[i] - x_avg,3)
    m_3 = temp3 / len(vyb)
    result_text += f'3-й центральний момент: {round(m_3, 4)}\n'
    temp4 = 0
    for i in range(len(xi)):
        temp4 += ni[i]*pow(xi[i] - x_avg, 4)
    m_4 = temp4 / len(vyb)
    result_text += f'4-й центральний момент: {round(m_4, 4)}\n\n'

    A_s = m_3/pow(m_2,3/2)
    asem = ''
    if A_s > 0:
        asem = 'cтатистичний матеріал скошений вправо'
    if A_s < 0:
        asem = 'cтатистичний матеріал скошений вліво'
    if A_s == 0:
        asem = 'cтатистичний матеріал симетричний відносно середини проміжку'

    result_text += f'Асиметрія: {round(A_s, 3)}, {asem}\n'

    E_k = (m_4/pow(m_2,2))-3
    eks = ''
    if E_k > 0:
        eks = 'cтатистичний матеріал – високовершинний'
    if E_k < 0:
        eks = 'cтатистичний матеріал – низьковершинний'
    if E_k == 0:
        eks = 'cтатистичний матеріал – нормальновершинний'
    result_text += f'Ексцес: {round(E_k, 3)}, {eks}\n'
    sample_text.insert(tk.END, result_text)


def frequency_table(vyb, sample_text):
    repet = Counter(vyb)
    chastoty = [repet[i] for i in sorted(repet)]
    xi = [item for item in sorted(repet)]

    sample_text.insert(tk.END, '\n\n---Частотна таблиця---\n')

    xi_text = 'Значення | Частота\n'
    for i, c in zip(xi, chastoty):
        xi_text += f'{i} | {c}\n'
    sample_text.insert(tk.END, xi_text)

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(xi, chastoty, color="green", width=0.1)
    axs[0].set_title('Діаграма частот')
    axs[0].set_xlabel('Значення')
    axs[0].set_ylabel('Частота значень')
    axs[0].set_xticks(xi)
    axs[0].set_yticks(chastoty)

    axs[1].plot(xi, chastoty, c="green")
    axs[1].set_title('Полігон частот')
    axs[1].set_xlabel('Значення')
    axs[1].set_ylabel('Частота значень')
    axs[1].set_xticks(xi)
    axs[1].set_yticks(chastoty)

    plt.show()
    return xi, chastoty


def emp_func(xi, ni, vybirka, sample_text):
    sample_text.insert(tk.END, '\n\n---Емпірична функція розподілу---\n')
    n = 0
    for i in range(len(ni)):
        n += ni[i]
        if i == 0:
            sample_text.insert(tk.END, f'\n0 , якщо x < {xi[i]}')
        elif i == len(ni) - 1:
            sample_text.insert(tk.END, f'\n{round(n/len(vybirka),2)}, якщо х > {xi[-1]}')
        else:
            sample_text.insert(tk.END, f'\n{round((n/len(vybirka)),3)},якщо {xi[i]} <= x < {xi[i+1]}')

    plt.title('Графік емпіричної функції')
    hist, edges = np.histogram(vybirka, bins=len(vybirka))
    Y = hist.cumsum() / 100
    for i in range(len(Y)):
        plt.plot([edges[i], edges[i + 1]], [Y[i], Y[i]], c="green")
    plt.xticks(xi)
    plt.show()


def generate_sample(size, start, end):
    if size < 50:
        raise ValueError("Розмір вибірки повинен бути не менше 50.")
    if start > end:
        raise ValueError("Початкове значення не може бути більшим за кінцеве значення.")

    data = [random.randint(start, end) for _ in range(size)]
    data.sort()
    return data


def generate_sample_click():
    try:
        if None in [sample_size_entry.get(), sample_start_entry.get(), sample_end_entry.get()]:
            raise ValueError("Заповніть усі три параметра.")

        size = int(sample_size_entry.get())
        start = int(sample_start_entry.get())
        end = int(sample_end_entry.get())
        vybirka = generate_sample(size, start, end)

        sample_window = tk.Toplevel(root)
        sample_window.title("Результат")
        sample_text = tk.Text(sample_window, height=100, width=300)
        sample_text.pack()
        sample_text.insert(tk.END, '\nВхідні дані:\n')
        sample_text.insert(tk.END, f'Кількість елементів: {size}\n')
        sample_text.insert(tk.END, f'Початок інтервалу: {start}\n')
        sample_text.insert(tk.END, f'Кінець інтервалу: {end}\n')

        sample_text.insert(tk.END, '\nЗадача 1\n')
        sample_text.insert(tk.END, f'Варіаційний ряд:\n{vybirka}\n')
        xi, ni = frequency_table(vybirka, sample_text)
        emp_func(xi, ni, vybirka, sample_text)
        harakter_dyskr(xi, ni, vybirka, sample_text)

        sample_text.insert(tk.END, '\nЗадача 2\n')
        inter_ser, w_i = interval_distr(xi, vybirka, sample_text)
        interval_emp_func(inter_ser, w_i, vybirka, sample_text)
        harakter_inter(inter_ser, w_i, vybirka, sample_text)

    except ValueError:
        error_window = tk.Toplevel(root)
        error_window.title("Помилка")
        text = tk.Label(error_window, text="Дані повинні бути числами")
        text.pack()


root = tk.Tk()
root.title("Статистичний аналіз")
sample_size_label = tk.Label(root, text="Розмір вибірки:")
sample_size_label.pack(pady=5)
sample_size_entry = tk.Entry(root)
sample_size_entry.pack(pady=5)
sample_range_label = tk.Label(root, text="Діапазон значень:")
sample_range_label.pack(pady=5)
start = tk.Label(root, text="Початок:")
start.pack(side="left", padx=5, pady=5)
sample_start_entry = tk.Entry(root, width=5)
sample_start_entry.pack(side="left", padx=5, pady=5)
sample_end_entry = tk.Entry(root, width=5)
sample_end_entry.pack(side="right", padx=5, pady=5)
end = tk.Label(root, text="Кінець:")
end.pack(side="right", padx=5, pady=5)
generate_sample_button = tk.Button(root, text="Згенерувати результат", command=generate_sample_click)
generate_sample_button.pack(side="bottom", padx=10, pady=10)

root.mainloop()
