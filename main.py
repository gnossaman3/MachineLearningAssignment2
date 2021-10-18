import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlrose_hiive
import time
from sklearn.model_selection import learning_curve

def get_data():
    df_income = pd.read_csv('Data/income_evaluation.csv', header=0, skipinitialspace=True)

    income_group_labels = {}
    for col in df_income.columns:
        income_group_labels[col] = {}
        if df_income[col].dtype == type(df_income['age']):
            i = 0
            for group in df_income[col].unique():
                income_group_labels[col][group] = i
                df_income[col] = df_income[col].replace(group, i)
                i += 1


    return df_income

if __name__ == '__main__':
    print('Beginning Project...')
    prob_sizes = [10, 100, 200, 500]

    print('Creating Fitness Functions...')
    fit1 = mlrose_hiive.OneMax()
    fit2 = mlrose_hiive.FlipFlop()
    fit3 = mlrose_hiive.FourPeaks()
    fitness_functions = [fit1, fit2, fit3]

    print('Testing Problems...')
    for size in prob_sizes:
        i = 0
        for fit in fitness_functions:
            print('Creating Problem ', i, ' With Problem Size ', size, '...')
            prob = mlrose_hiive.DiscreteOpt(length=size, fitness_fn=fit)

            print('Testing Random Hill Climbing...')
            t1 = time.time()
            best_state, best_fitness, hill_fitness_curve = mlrose_hiive.random_hill_climb(problem=prob, max_iters=2000,
                                                                                          curve=True)
            hill_time = time.time() - t1
            print('Testing Simulated Annealing...')
            t2 = time.time()
            best_state, best_fitness, sim_fitness_curve = mlrose_hiive.simulated_annealing(problem=prob, max_iters=2000,
                                                                                           curve=True)
            sim_time = time.time() - t2
            print('Testing Genetic Algorithm...')
            t3 = time.time()
            best_state, best_fitness, genetic_fitness_curve = mlrose_hiive.genetic_alg(problem=prob, max_iters=2000,
                                                                                       curve=True)
            gen_time = time.time() - t3
            print('Testing MIMIC...')
            t4 = time.time()
            best_state, best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(problem=prob, max_iters=2000, curve=True)
            mimic_time = time.time() - t4

            print('Fitness Functions For Problem ', i, ' and Problem Size ', size, ' Complete. Generating Graphs...')
            title = 'Problem ' + str(i) + ' Fitness Scores Using Problem Size ' + str(size)
            fname = 'Problem' + str(i) + 'FitnessScores' + str(size)
            plt.plot(np.arange(len(hill_fitness_curve[:,0])), hill_fitness_curve[:,0], label='Random Hill Climbing')
            plt.plot(np.arange(len(sim_fitness_curve[:,0])), sim_fitness_curve[:,0], label='Simulated Annealing')
            plt.plot(np.arange(len(genetic_fitness_curve[:,0])), genetic_fitness_curve[:,0], label='Genetic Algorithm')
            plt.plot(np.arange(len(mimic_fitness_curve[:,0])), mimic_fitness_curve[:,0], label='MIMIC')
            plt.title(title)
            plt.xlabel('Iterations')
            plt.ylabel('Fitness Score')
            plt.legend(loc='best')
            plt.savefig(fname=fname)
            plt.close()

            print('Graphs Created. Printing Wall Clock Time For Problem ', i, ' and Problem Size', size, '...')
            print()
            print('          | Random Hill Climbing | Simulated Annealing | Genetic Algorithm | MIMIC')
            print('---------------------------------------------------------------------------------------')
            print('Problem ', i,':|     ', hill_time, 's       |        ', sim_time, 's       |        ', gen_time, 's        |   ', mimic_time, 's')
            i += 1

    print('Completed First Portion of Project. Moving on to NN...')
    print('Preparing Data...')
    data = get_data()
    data_X = data.loc[:, data.columns != 'income']
    data_Y = data['income']

    print('Data Prepared. Creating Backprop NN...')
    hidden_nodes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    backprop_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='tanh', algorithm='gradient_descent',
                                             max_iters=200, learning_rate=0.0001)
    print('Done. Creating Random Hill NN...')
    hill_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='tanh',
                                              algorithm='random_hill_climb', max_iters=200, learning_rate=0.0001)
    print('Done. Creating Simulated Annealing NN...')
    sim_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='tanh', algorithm='simulated_annealing',
                                        max_iters=200, learning_rate=0.0001)
    print('Done. Creating Genetic Algorithm NN...')
    gen_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, activation='tanh', algorithm='genetic_alg',
                                        max_iters=200, learning_rate=0.0001)

    print('Done. Training Backprop NN...')
    train_sizes = np.linspace(.1, 1.0, 10)
    backprop_train_sizes, backprop_train_scores, backprop_test_scores, backprop_fit_times, backprop_score_times = \
        learning_curve(backprop_nn, data_X, data_Y, train_sizes=train_sizes, cv=2, return_times=True, verbose=5)

    print('Done. Training Random Hill Climbing NN...')
    hill_train_sizes, hill_train_scores, hill_test_scores, hill_fit_times, hill_score_times = \
        learning_curve(hill_nn, data_X, data_Y, train_sizes=train_sizes, cv=2, return_times=True, verbose=5)

    print('Done. Training Simulated Annealing NN...')
    sim_train_sizes, sim_train_scores, sim_test_scores, sim_fit_times, sim_score_times = \
        learning_curve(sim_nn, data_X, data_Y, train_sizes=train_sizes, cv=2, return_times=True, verbose=5)

    print('Done. Training Genetic Algorithm NN...')
    gen_train_sizes, gen_train_scores, gen_test_scores, gen_fit_times, gen_score_times = \
        learning_curve(gen_nn, data_X, data_Y, train_sizes=train_sizes, cv=2, return_times=True, verbose=5)

    backprop_train_scores = np.mean(backprop_train_scores, axis=1)
    backprop_test_scores = np.mean(backprop_test_scores, axis=1)
    backprop_fit_times = np.mean(backprop_fit_times, axis=1)
    hill_train_scores = np.mean(hill_train_scores, axis=1)
    hill_test_scores = np.mean(hill_test_scores, axis=1)
    hill_fit_times = np.mean(hill_fit_times, axis=1)
    sim_train_scores = np.mean(sim_train_scores, axis=1)
    sim_test_scores = np.mean(sim_test_scores, axis=1)
    sim_fit_times = np.mean(sim_fit_times, axis=1)
    gen_train_scores = np.mean(gen_train_scores, axis=1)
    gen_test_scores = np.mean(gen_test_scores, axis=1)
    gen_fit_times = np.mean(gen_fit_times, axis=1)

    print('Done. Creating Backprop Graph...')
    plt.plot(backprop_train_sizes, backprop_train_scores, label='Training Score')
    plt.plot(backprop_train_sizes, backprop_test_scores, label='Testing Score')
    plt.title('Learning Curve for NN using Gradient Descent')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='BackpropLearningCurve')
    plt.close()

    print('Done. Creating Random Hill Climbing Graph...')
    plt.plot(hill_train_sizes, hill_train_scores, label='Traning Score')
    plt.plot(hill_train_sizes, hill_test_scores, label='Testing Scores')
    plt.title('Learning Curve for NN using Random Hill Climbing')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='HillLearningCurve')
    plt.close()

    print('Done. Creating Simulated Annealing Graph...')
    plt.plot(sim_train_sizes, sim_train_scores, label='Training Scores')
    plt.plot(sim_train_sizes, sim_test_scores, label='Testing Scores')
    plt.title('Learning Curve for NN using Simulated Annealing')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='SimLearningCurve')
    plt.close()

    print('Done. Creating Genetic Algorithm Graph...')
    plt.plot(gen_train_sizes, gen_train_scores, label='Training Scores')
    plt.plot(gen_train_sizes, gen_test_scores, label='Testing Scores')
    plt.title('Learning Curve for NN using Genetic Algorithm')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='GenLearningCurve')
    plt.close()

    print('Done. Creating Time Graph...')
    plt.plot(backprop_train_sizes, backprop_fit_times, label='Gradient Descent')
    plt.plot(hill_train_sizes, hill_fit_times, label='Random Hill Climbing')
    plt.plot(sim_train_sizes, sim_fit_times, label='Simulated Annealing')
    plt.plot(gen_train_sizes, gen_fit_times, label='Genetic Algorithm')
    plt.title('Learning Curve For Fit Times Of Varying Backprop Algorithms')
    plt.xlabel('Training Size')
    plt.ylabel('Fit Times')
    plt.legend(loc='best')
    plt.savefig(fname='TimesLearningCurve')
    plt.close()

    print('Done.')