import numpy as np
import copy
import matplotlib.pyplot as plt


def show_q(q_matrix):
    val1 = [i for i in range(25)]
    val2 = val1
    val3 = q_matrix
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table_q = ax.table(
        cellText=val3,
        rowLabels=val2,
        colLabels=val1,
        rowColours=["palegreen"] * 25,
        colColours=["palegreen"] * 25,
        cellLoc='center',
        loc='best')
    table_q.scale(1, 0.95)
    table_q.auto_set_font_size(False)
    table_q.set_fontsize(7)
    ax.set_title('Q matrix', fontweight="bold", fontsize=20)
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_axis_off()
    val1 = [[i for i in range(5)], [i for i in range(5, 10)], [i for i in range(10, 15)], [i for i in range(15, 20)], [i for i in range(20, 25)]]
    table_q = ax.table(
        cellText=val1,
        cellLoc='center',
        loc='best')
    table_q.scale(1, 4.95)
    plt.show()
    plt.close()


def q_learning(r_matrix, q_matrix=None, goal_position=0, gamma=0.8, alpha=0.01, num_episode=1000000, min_difference=1e-3, typef='max'):
    if typef == 'max':
        print("\nQ-learning with choosing max value")
    else:
        print("\nQ-learning with choosing random value")
    if num_episode != 1:
        q_matrix = np.zeros(r_matrix.shape)
    all_position = np.arange(len(r_matrix))
    for i in range(num_episode):
        q_last = copy.deepcopy(q_matrix)
        initial_position = np.random.choice(all_position)
        if typef == 'max':
            possible_steps = np.where(r_matrix[initial_position] != -1)[0]
            best_step = np.where(q_matrix[initial_position] == max([q_matrix[initial_position][i] for i in np.where(r_matrix[initial_position] != -1)[0]]))[0]
            common = list(set(possible_steps) & set(best_step))
            action = np.random.choice(common)
        else:
            action = np.random.choice(np.where(r_matrix[initial_position] != -1)[0])
        q_matrix[initial_position][action] = q_matrix[initial_position][action] + alpha * \
                                             (r_matrix[initial_position][action] + gamma * np.max(q_matrix[action]) -
                                              q_matrix[initial_position][action])
        current_position = action
        if num_episode == 1:
            print("Changing position:")
            print("Start:", initial_position)
        counter = 1
        while current_position != goal_position:
            if num_episode == 1:
                print("Step", counter, "->", current_position)
            if typef == 'max':
                possible_steps = np.where(r_matrix[current_position] != -1)[0]
                best_step = np.where(q_matrix[current_position] == max([q_matrix[current_position][i] for i in np.where(r_matrix[current_position] != -1)[0]]))[0]
                common = list(set(possible_steps) & set(best_step))
                action = np.random.choice(common)
            else:
                action = np.random.choice(np.where(r_matrix[current_position] != -1)[0])
            q_matrix[current_position][action] = q_matrix[current_position][action] + alpha * (
                    r_matrix[current_position][action] + gamma * np.max(q_matrix[action]) -
                    q_matrix[current_position][action])
            current_position = action
            counter += 1
        diff = np.sum(q_matrix - q_last)
        if diff < min_difference:
            if num_episode != 1:
                print('Number of epoch', i)
            else:
                print("Step", counter, "->", goal_position)
            break
    return np.around(q_matrix / np.max(q_matrix) * 100)


# def q_learning_max(r_matrix, q_matrix=None, goal_position=0, gamma=0.8, alpha=0.01, num_episode=1000000, min_difference=1e-5):
#     print("\nQ-learning with choosing max value")
#     if num_episode != 1:
#         q_matrix = np.zeros(r_matrix.shape)
#     all_position = np.arange(len(r_matrix))
#     for i in range(num_episode):
#         q_last = copy.deepcopy(q_matrix)
#         initial_position = np.random.choice(all_position)
#         possible_steps = np.where(r_matrix[initial_position] != -1)[0]
#         best_step = np.where(q_matrix[initial_position] == max([q_matrix[initial_position][i] for i in np.where(r_matrix[initial_position] != -1)[0]]))[0]
#         common = list(set(possible_steps) & set(best_step))
#         action = np.random.choice(common)
#         q_matrix[initial_position][action] = q_matrix[initial_position][action] + alpha * \
#                                              (r_matrix[initial_position][action] + gamma * np.max(q_matrix[action]) -
#                                               q_matrix[initial_position][action])
#         current_position = action
#         if num_episode == 1:
#             print("Changing position:")
#             print(initial_position)
#         while current_position != goal_position:
#             if num_episode == 1:
#                 print(current_position)
#             possible_steps = np.where(r_matrix[current_position] != -1)[0]
#             best_step = np.where(q_matrix[current_position] == max([q_matrix[current_position][i] for i in np.where(r_matrix[current_position] != -1)[0]]))[0]
#             common = list(set(possible_steps) & set(best_step))
#             action = np.random.choice(common)
#             q_matrix[current_position][action] = q_matrix[current_position][action] + alpha * (
#                     r_matrix[current_position][action] + gamma * np.max(q_matrix[action]) -
#                     q_matrix[current_position][action])
#             current_position = action
#         diff = np.sum(q_matrix - q_last)
#         if diff < min_difference:
#             if num_episode != 1:
#                 print('Number of epoch', i)
#             else:
#                 print(goal_position)
#             break
#     return np.around(q_matrix / np.max(q_matrix) * 100)


def run():
    r_matrix = np.array([[-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                         [100, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1],
                         [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1],
                         [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                         [-1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1],
                         [100, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1],
                         [-1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1],
                         [-1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0],
                         [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1],
                         [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
                         [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1],
                         [-1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0],
                         [-1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1]])
    q_matrix1 = q_learning(r_matrix, typef='max').astype(int)
    show_q(q_matrix1)
    q_learning(r_matrix, q_matrix1, num_episode=1, typef='max')

    q_matrix2 = q_learning(r_matrix, typef='rand').astype(int)
    show_q(q_matrix2)
    q_learning(r_matrix, q_matrix2, num_episode=1, typef='rand')


run()
