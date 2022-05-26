from math import exp

speed = 0.1
data = [0.13, 5.97, 0.57, 4.02, 0.31, 5.55, 0.15, 4.54, 0.65, 4.34, 1.54, 4.70, 0.58, 5.83, 0.03]
weights = [1, 1, 1]


def activation(s):
    return (1 / (1 + exp(-s))) * 10


def squared_error(y_actual, y_expected):
    return (y_actual - y_expected) ** 2


def error(y_actual, y_expected, s, x):
    return [(y_actual - y_expected) * (exp(-s) / (1 + exp(-s)) ** 2) * xi for xi in x]


def new_weights(errors):
    return [-speed * error_i for error_i in errors]


def studying():
    global data, speed, weights
    sum_squared_errors, sum_squared_errors0, counter = 0.0, 0.0, 0

    while True:
        result = []
        all_w = [[], [], []]

        for i in range(len(data) - 5):
            x = [data[i], data[i + 1], data[i + 2]]
            y_expected = data[i + 3]
            s = sum(wi * xi for wi, xi in zip(weights, x))
            y_actual = activation(s)
            result.append(y_actual)
            sum_squared_errors += squared_error(y_actual, y_expected)
            w = new_weights(error(y_actual, y_expected, s, x))

            for j in range(len(w)):
                all_w[j].append(w[j])

        for k in range(len(all_w)):
            weights[k] += (sum(all_w[k]) / len(all_w[k]))

        diff = abs(sum_squared_errors-sum_squared_errors0)
        counter += 1
        if diff < 0.0001 or counter > 100000:
            return result


def testing():
    result = []
    for i in range(10, 12):
        x = [data[i], data[i + 1], data[i + 2]]
        result.append(activation(sum([wi * xi for wi, xi in zip(weights, x)])))
    return result


def print_results(studying_result, testing_result):
    print('Studying speed:', speed, '\nStudying data:', data)

    print('\nStudying\n', '{:<22}{:<22}{:<15}{:<22}'.format('Input', 'Actual', 'Expected', 'Error'), sep='')
    for i in range(len(data) - 5):
        x = [data[i], data[i + 1], data[i + 2]]
        print("{:<22}{:<22}{:<15}{:<22}".format(str(x), studying_result[i], data[i + 3],
                                                abs(data[i + 3] - studying_result[i])))
    print('\nWeights:', weights)

    print('\nTesting\n', '{:<22}{:<22}{:<15}{:<22}'.format('Input', 'Actual', 'Expected', 'Error'), sep='')
    for i, j in zip(range(10, 12), range(2)):
        x = [data[i], data[i + 1], data[i + 2]]
        print("{:<22}{:<22}{:<15}{:<22}".format(str(x), testing_result[j], data[i + 3],
                                                abs(data[i + 3] - testing_result[j])))


print_results(studying(), testing())
