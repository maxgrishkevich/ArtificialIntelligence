def or_and_not(values, t, weights):
    for value in values:
        output = 0
        for i in range(len(value) - 1):
            output += weights[i] * value[i]
        output = 1 if output >= t else 0
        print('{:<10}{:<10}{:<10}'.format(str(value[:-1]), output, value[-1]))


def xor(values, weights1, weights2):
    for value in values:
        output = 0
        for i in range(len(value[:-1])):
            s = 0
            for j in range(len(weights1)):
                s += value[j] * weights1[i][j]
            output += 1 * weights2[i] if s >= 0.5 else 0 * weights2[i]
        output = 1 if output >= 0.5 else 0
        print('{:<10}{:<10}{:<10}'.format(str(value[:-1]), output, value[-1]))


print('{}{:<10}{:<10}{:<10}'.format('OR\n', 'Input', 'Actual', 'Expected'))
or_and_not([[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]], 0.5, [1, 1])

print('{}{:<10}{:<10}{:<10}'.format('\nAND\n', 'Input', 'Actual', 'Expected'))
or_and_not([[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]], 1.5, [1, 1])

print('{}{:<10}{:<10}{:<10}'.format('\nNOT\n', 'Input', 'Actual', 'Expected'))
or_and_not([[0, 1],
            [1, 0]], -1, [-1.5])

print('{}{:<10}{:<10}{:<10}'.format('\nXOR\n', 'Input', 'Actual', 'Expected'))
xor([[0, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]], [[1, -1], [-1, 1]], [1, 1])
