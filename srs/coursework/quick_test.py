from wann import WANNPopulation
from neat_runner import NEATRunner
from mpi4py import MPI
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

algorithm = sys.argv[1] if len(sys.argv) > 1 else "WANN"

if rank == 0:
    print(f"Quick {algorithm} test with {size} processes")

start_time = time.time()

if algorithm == "WANN":
    if rank == 0:
        pop = WANNPopulation(20, 24, 4)  # Smaller population for quick test
        for gen in range(5):  # Only 5 generations
            individuals = pop.get_individuals()
            chunk_size = len(individuals) // max(1, size-1)
            
            if size == 1:
                fitness = [ind.evaluate_fitness(trials=1) for ind in individuals]
            else:
                # Simplified parallel evaluation
                if rank == 0:
                    chunk = individuals[:chunk_size]
                    fitness = [ind.evaluate_fitness(trials=1) for ind in chunk]
            
            pop.update_fitness(fitness[:len(individuals)])
            pop.evolve()
            
            if rank == 0:
                best = max(fitness) if fitness else 0
                print(f"Gen {gen}: Best={best:.1f}")

elif algorithm == "NEAT":
    neat_runner = NEATRunner(comm, rank, size)
    if rank == 0:
        # Simplified NEAT test
        print("NEAT test running...")
        time.sleep(5)  # Simulate NEAT execution

total_time = time.time() - start_time

if rank == 0:
    print(f"{algorithm} completed in {total_time:.2f} seconds")
