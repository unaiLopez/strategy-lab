import warnings
warnings.filterwarnings('ignore')

from genopt.environment import Environment
from genopt.parameters import Parameters
from strategy import Strategy

class StrategyOptimizer:
    def __init__(self, ticker: str, months_validation: int):
        self.ticker = ticker
        self.months_validation = months_validation
        self.strategy = Strategy()

    def _define_search_space(self) -> dict:
        params = {
            'interval': Parameters.suggest_categorical(['5M', '15M', '30M', '1H', '2H', '4H', '6H', '12H']),
            'take_profit': Parameters.suggest_float(0.005, 0.1),
            'stop_loss': Parameters.suggest_float(0.005, 0.1),
            'velocity_up': Parameters.suggest_float(0.01,1.0),
            'velocity_down': Parameters.suggest_float(-0.01,-1.0),
            'acceleration_up': Parameters.suggest_float(0.01,1.0),
            'acceleration_down': Parameters.suggest_float(-0.01,-1.0)
        }

        return params
    
    def objective(self, individual: dict) -> float:
        interval = individual['interval']
        take_profit = individual['take_profit']
        stop_loss = individual['stop_loss']
        velocity_up = individual['velocity_up']
        velocity_down = individual['velocity_down']
        acceleration_up = individual['acceleration_up']
        acceleration_down = individual['acceleration_down']
        
        mean_rois = self.strategy.apply_strategy(self.ticker, interval, take_profit, stop_loss, self.months_validation,
                                    velocity_up, velocity_down, acceleration_up, acceleration_down)
        
        return mean_rois

    def optimize(self, timeout: int, n_jobs: int, num_generations: int, num_population: int) -> dict:
        params = self._define_search_space()
        environment = Environment(
            params=params,
            num_population=num_population,
            selection_type='ranking',
            selection_rate=0.8,
            crossover_type='two-point',
            mutation_type='single-gene',
            prob_mutation=0.25,
            verbose=1,
            random_state=42
        )

        results = environment.optimize(
            objective=self.objective,
            direction='maximize',
            num_generations=num_generations,
            timeout=timeout,
            n_jobs=n_jobs
        )

        return results.best_individual
    
