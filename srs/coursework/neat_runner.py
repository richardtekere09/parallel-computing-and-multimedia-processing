import neat
import gymnasium as gym
import numpy as np
import pickle
from mpi4py import MPI
from typing import List, Tuple
import logging

class NEATRunner:
    """Parallel NEAT evolution runner"""
    
    def __init__(self, comm, rank: int, size: int):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.logger = logging.getLogger(f'neat_rank_{rank}')
        
        # NEAT configuration
        self.config = neat.Config(neat.DefaultGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  'config/neat_config.txt')
    
    def evaluate_genome(self, genome, config) -> float:
        """Evaluate a single NEAT genome"""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        env = gym.make('BipedalWalker-v3')
        total_reward = 0
        trials = 3
        
        for trial in range(trials):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(1600):
                action = net.activate(obs)
                action = np.clip(action, -1, 1)
                
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        env.close()
        return total_reward / trials
    
    def parallel_evaluate(self, genomes, config):
        """Parallel evaluation of genome population"""
        if self.rank == 0:
            # Master process: distribute genomes
            genome_list = list(genomes)
            chunk_size = len(genome_list) // self.size
            
            # Send genome chunks to workers
            for i in range(1, self.size):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.size - 1 else len(genome_list)
                chunk = genome_list[start_idx:end_idx]
                self.comm.send(chunk, dest=i, tag=0)
            
            # Evaluate master's chunk
            master_chunk = genome_list[:chunk_size]
            for genome_id, genome in master_chunk:
                genome.fitness = self.evaluate_genome(genome, config)
            
            # Receive results from workers
            for i in range(1, self.size):
                worker_results = self.comm.recv(source=i, tag=1)
                for genome_id, fitness in worker_results:
                    # Find and update genome
                    for gid, g in genomes:
                        if gid == genome_id:
                            g.fitness = fitness
                            break
        
        else:
            # Worker process: evaluate received genomes
            genome_chunk = self.comm.recv(source=0, tag=0)
            results = []
            
            for genome_id, genome in genome_chunk:
                fitness = self.evaluate_genome(genome, config)
                results.append((genome_id, fitness))
            
            self.comm.send(results, dest=0, tag=1)
    
    def run_evolution(self, generations: int = 50):
        """Run NEAT evolution with parallel evaluation"""
        if self.rank == 0:
            # Master process manages evolution
            population = neat.Population(self.config)
            
            # Add reporters
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            
            # Custom evaluation function
            def eval_genomes(genomes, config):
                self.parallel_evaluate(genomes, config)
            
            # Run evolution
            best_genome = population.run(eval_genomes, generations)
            
            # Save results
            with open('results/neat_best_genome.pkl', 'wb') as f:
                pickle.dump(best_genome, f)
            
            # Save fitness statistics
            fitness_history = [stats.get_fitness_mean() for _ in range(generations)]
            np.savetxt('results/neat_fitness.csv', fitness_history, delimiter=',')
            
            self.logger.info(f"NEAT evolution completed. Best fitness: {best_genome.fitness}")
        
        else:
            # Worker processes participate in evaluation
            for generation in range(generations):
                self.parallel_evaluate([], self.config)  # Receive and evaluate