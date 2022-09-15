import warnings
warnings.filterwarnings('ignore')

from optimizer import StrategyOptimizer
from strategy import Strategy

if __name__ == '__main__':
    ticker = 'BTCEUR'
    months_validation = 3
    timeout = 1500
    n_jobs = -1
    num_generations = 999999
    num_population = 120

    strategy_optimizer = StrategyOptimizer(ticker=ticker, months_validation=months_validation)
    best_individual = strategy_optimizer.optimize(timeout, n_jobs, num_generations, num_population)

    interval = best_individual['interval']
    take_profit = best_individual['take_profit']
    stop_loss = best_individual['stop_loss']
    velocity_up = best_individual['velocity_up']
    velocity_down = best_individual['velocity_down']
    acceleration_up = best_individual['acceleration_up']
    acceleration_down = best_individual['acceleration_down']

    strategy = Strategy()
    strategy.plot_train_test_folds_performance(
                                               ticker, interval, take_profit, stop_loss, months_validation,
                                               velocity_up, velocity_down, acceleration_up,
                                               acceleration_down
    )
