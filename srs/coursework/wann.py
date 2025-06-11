import numpy as np
import gym
import networkx as nx
from typing import List, Tuple
import random
try:
    from opencl_utils import OpenCLAccelerator
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

class WANNGenome:
    """Weight Agnostic Neural Network Genome"""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}  # node_id: (layer, activation_func)
        self.connections = {}  # (src, dst): enabled
        self.fitness = 0.0
        
        # Initialize minimal topology
        self._initialize_minimal_topology()
    
    def _initialize_minimal_topology(self):
        """Create minimal feed-forward topology"""
        # Input nodes
        for i in range(self.input_size):
            self.nodes[i] = (0, 'linear')  # layer 0, linear activation
        
        # Output nodes
        for i in range(self.output_size):
            node_id = self.input_size + i
            self.nodes[node_id] = (1, 'tanh')  # layer 1, tanh activation
            
            # Connect all inputs to all outputs
            for j in range(self.input_size):
                self.connections[(j, node_id)] = True
    
    def add_node(self):
        """Add a new hidden node by splitting an existing connection"""
        if not self.connections:
            return
        
        # Select random enabled connection
        enabled_connections = [conn for conn, enabled in self.connections.items() if enabled]
        if not enabled_connections:
            return
        
        src, dst = random.choice(enabled_connections)
        
        # Create new node
        new_node_id = max(self.nodes.keys()) + 1
        src_layer = self.nodes[src][0]
        dst_layer = self.nodes[dst][0]
        new_layer = (src_layer + dst_layer) / 2
        
        self.nodes[new_node_id] = (new_layer, random.choice(['tanh', 'relu', 'sigmoid']))
        
        # Disable original connection and add two new ones
        self.connections[(src, dst)] = False
        self.connections[(src, new_node_id)] = True
        self.connections[(new_node_id, dst)] = True
    
    def add_connection(self):
        """Add a new connection between existing nodes"""
        node_ids = list(self.nodes.keys())
        
        for _ in range(10):  # Try 10 times
            src = random.choice(node_ids)
            dst = random.choice(node_ids)
            
            # Check if connection is valid (no cycles, different layers)
            if (src != dst and 
                self.nodes[src][0] < self.nodes[dst][0] and 
                (src, dst) not in self.connections):
                
                self.connections[(src, dst)] = True
                break
    
    def forward_pass(self, inputs: np.ndarray, shared_weight: float = 1.0) -> np.ndarray:
        """Execute forward pass with shared weight"""
        if OPENCL_AVAILABLE and hasattr(self, 'cl_accelerator'):
            return self._forward_pass_opencl(inputs, shared_weight)
        else:
            return self._forward_pass_cpu(inputs, shared_weight)
    
    def _forward_pass_cpu(self, inputs: np.ndarray, shared_weight: float) -> np.ndarray:
        """CPU-based forward pass"""
        node_values = {}
        
        # Set input values
        for i, val in enumerate(inputs):
            if i < self.input_size:
                node_values[i] = val
        
        # Process nodes in layer order
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1][0])
        
        for node_id, (layer, activation) in sorted_nodes:
            if node_id < self.input_size:  # Skip input nodes
                continue
            
            # Sum weighted inputs
            total_input = 0.0
            for (src, dst), enabled in self.connections.items():
                if dst == node_id and enabled and src in node_values:
                    total_input += node_values[src] * shared_weight
            
            # Apply activation function
            if activation == 'tanh':
                node_values[node_id] = np.tanh(total_input)
            elif activation == 'relu':
                node_values[node_id] = max(0, total_input)
            elif activation == 'sigmoid':
                node_values[node_id] = 1 / (1 + np.exp(-np.clip(total_input, -500, 500)))
            else:  # linear
                node_values[node_id] = total_input
        
        # Extract output values
        outputs = []
        for i in range(self.output_size):
            output_id = self.input_size + i
            outputs.append(node_values.get(output_id, 0.0))
        
        return np.array(outputs)
    
    def evaluate_fitness(self, trials: int = 3) -> float:
        """Evaluate fitness across multiple weight values"""
        env = gym.make('BipedalWalker-v3')
        
        # Test with different shared weights
        weight_values = [-2, -1, -0.5, 0.5, 1, 2]
        total_reward = 0
        
        for weight in weight_values:
            for trial in range(trials):
                obs = env.reset()
                episode_reward = 0
                
                for step in range(1600):  # Max episode length
                    action = self.forward_pass(obs, weight)
                    action = np.clip(action, -1, 1)  # Clip to valid range
                    
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                
                total_reward += episode_reward
        
        env.close()
        self.fitness = total_reward / (len(weight_values) * trials)
        return self.fitness

class WANNPopulation:
    """Population manager for WANN evolution"""
    
    def __init__(self, pop_size: int, input_size: int, output_size: int):
        self.pop_size = pop_size
        self.input_size = input_size
        self.output_size = output_size
        self.population = []
        self.generation = 0
        
        # Initialize population
        for _ in range(pop_size):
            genome = WANNGenome(input_size, output_size)
            self.population.append(genome)
        
        # Setup OpenCL if available
        if OPENCL_AVAILABLE:
            try:
                self.cl_accelerator = OpenCLAccelerator()
                for genome in self.population:
                    genome.cl_accelerator = self.cl_accelerator
            except Exception as e:
                print(f"OpenCL initialization failed: {e}")
    
    def get_individuals(self) -> List[WANNGenome]:
        """Get current population for evaluation"""
        return self.population
    
    def update_fitness(self, fitness_scores: List[float]):
        """Update fitness scores for population"""
        for i, fitness in enumerate(fitness_scores):
            if i < len(self.population):
                self.population[i].fitness = fitness
    
    def evolve(self):
        """Evolve population for next generation"""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep top 20% as parents
        num_parents = max(1, self.pop_size // 5)
        parents = self.population[:num_parents]
        
        # Generate new population
        new_population = parents.copy()  # Elitism
        
        while len(new_population) < self.pop_size:
            parent = random.choice(parents)
            child = self._mutate(parent)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def _mutate(self, parent: WANNGenome) -> WANNGenome:
        """Create mutated copy of parent"""
        child = WANNGenome(self.input_size, self.output_size)
        child.nodes = parent.nodes.copy()
        child.connections = parent.connections.copy()
        
        # Mutation probabilities
        if random.random() < 0.3:  # Add node
            child.add_node()
        
        if random.random() < 0.5:  # Add connection
            child.add_connection()
        
        if random.random() < 0.1:  # Toggle connection
            if child.connections:
                conn = random.choice(list(child.connections.keys()))
                child.connections[conn] = not child.connections[conn]
        
        return child