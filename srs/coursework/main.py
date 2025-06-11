#!/usr/bin/env python3
import numpy as np
import time
import os
from mpi4py import MPI
import logging
from wann import WANNPopulation
from neat_runner import NEATRunner
from visualization import NetworkVisualizer

def setup_logging(rank):
    """Setup logging for each MPI process"""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f'neuroevo_rank_{rank}')
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(f'{log_dir}/rank_{rank}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def run_wann_experiment(comm, rank, size, logger):
    """Run WANN parallel experiment"""
    logger.info(f"Starting WANN experiment on rank {rank}")
    
    if rank == 0:
        # Master process
        population = WANNPopulation(pop_size=100, input_size=24, output_size=4)
        best_fitness_history = []
        start_time = time.time()
        
        for generation in range(50):
            gen_start = time.time()
            
            # Distribute population for evaluation
            individuals = population.get_individuals()
            chunk_size = len(individuals) // size
            
            # Scatter individuals to workers
            for i in range(1, size):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < size - 1 else len(individuals)
                comm.send(individuals[start_idx:end_idx], dest=i, tag=0)
            
            # Evaluate master's chunk
            master_chunk = individuals[:chunk_size]
            master_fitness = [ind.evaluate_fitness() for ind in master_chunk]
            
            # Gather fitness results
            all_fitness = master_fitness
            for i in range(1, size):
                worker_fitness = comm.recv(source=i, tag=1)
                all_fitness.extend(worker_fitness)
            
            # Update population
            population.update_fitness(all_fitness)
            best_fitness = max(all_fitness)
            best_fitness_history.append(best_fitness)
            
            gen_time = time.time() - gen_start
            logger.info(f"Generation {generation}: Best={best_fitness:.3f}, Time={gen_time:.2f}s")
            
            # Evolve population
            population.evolve()
        
        total_time = time.time() - start_time
        logger.info(f"WANN experiment completed in {total_time:.2f} seconds")
        
        # Save results
        np.savetxt('results/wann_fitness.csv', best_fitness_history, delimiter=',')
        
    else:
        # Worker process
        for generation in range(50):
            # Receive individuals to evaluate
            individuals = comm.recv(source=0, tag=0)
            
            # Evaluate fitness
            fitness_scores = []
            for ind in individuals:
                fitness = ind.evaluate_fitness()
                fitness_scores.append(fitness)
            
            # Send results back
            comm.send(fitness_scores, dest=0, tag=1)

def run_neat_experiment(comm, rank, size, logger):
    """Run NEAT parallel experiment"""
    logger.info(f"Starting NEAT experiment on rank {rank}")
    
    neat_runner = NEATRunner(comm, rank, size)
    neat_runner.run_evolution(generations=50)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    logger = setup_logging(rank)
    
    if rank == 0:
        print(f"Running parallel neuroevolution with {size} processes")
        print("MacBook Pro 2018 - Intel Core i5")
    
    # Run experiments
    run_wann_experiment(comm, rank, size, logger)
    
    comm.Barrier()  # Synchronize before NEAT
    
    run_neat_experiment(comm, rank, size, logger)
    
    if rank == 0:
        # Generate visualizations
        visualizer = NetworkVisualizer()
        visualizer.plot_fitness_curves()
        visualizer.plot_performance_scaling()

if __name__ == "__main__":
    main()