from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

bounds = [[0.00, 5.0]]
iter_num = 100
bits = 16
population = 1000
cross_per = 0.9
mut_per = 0.1


def func(x):
    return x[0]**(1/2) * np.sin(10*x[0])


def decoding(boundsf, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(boundsf)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = boundsf[i][0] + (integer / largest) * (boundsf[i][1] - boundsf[i][0])
        decoded.append(value)
    return decoded


def selection(pop, values, k=3):
    sel_x = randint(len(pop))
    for x in randint(0, len(pop), k - 1):
        if values[x] < values[sel_x]:
            sel_x = x
    return pop[sel_x]


def crossover(p1, p2, cross_perf):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < cross_perf:
        s = randint(1, len(p1) - 2)
        c1 = p1[:s] + p2[s:]
        c2 = p2[:s] + p1[s:]
    return [c1, c2]


def mutation(bitstring, mut_perf):
    for i in range(len(bitstring)):
        if rand() < mut_perf:
            bitstring[i] = 1 - bitstring[i]



def ga_min_max(function, boundsf, bitsf, iter_numf, populationf, cross_perf, mut_perf, type_func):
    pop = [randint(0, 2, bitsf * len(boundsf)).tolist() for _ in range(populationf)]
    x, y = 0, function(decoding(boundsf, bitsf, pop[0]))
    values = []
    p1, p2 = 0, 0
    for _ in range(iter_numf):
        decoded_x = [decoding(boundsf, bitsf, p) for p in pop]
        values = [function(d) for d in decoded_x]
    for i in range(populationf):
        if type_func == 'max':
            if values[i] >= y:
                x, y = pop[i], values[i]
        elif type_func == 'min':
            if values[i] <= y:
                x, y = pop[i], values[i]
    selected = [selection(pop, values) for _ in range(populationf)]
    children = list()
    for i in range(0, populationf, 2):
        p1, p2 = selected[i], selected[i+1]
    for c in crossover(p1, p2, cross_perf):
        mutation(c, mut_perf)
        children.append(c)
    return [x, y]


def main():
    x_min, y_min = ga_min_max(func, bounds, bits, iter_num, population, cross_per, mut_per, 'min')
    x_max, y_max = ga_min_max(func, bounds, bits, iter_num, population, cross_per, mut_per, 'max')
    decoded_min = decoding(bounds, bits, x_min)
    print('min: f(%s) = %f' % (decoded_min, y_min))
    decoded_max = decoding(bounds, bits, x_max)
    print('max: f(%s) = %f' % (decoded_max, y_max))

    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 1000)
    y = x**(1/2) * np.sin(10*x)
    ax.plot(x, y)
    ax.scatter([decoded_min, decoded_max], [y_min, y_max], c='red', s=20)
    plt.show()


main()
