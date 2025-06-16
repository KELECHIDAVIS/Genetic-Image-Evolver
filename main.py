
'''
Evolve Agents to recreate image over generations
their genome is going to be the shapes they have
the fitness is going to be the mean squared error from the target image (lower is better)

Initially:
Parent genome is blank list of mutations , the fitness is then calculated 

Repeat: 
create child agents from adding random circle to parent genome 
calculate fitness of child agents
if child fitness is better than parent fitness, change parent genome to best child genome 
'''
import numpy as np
import random
from Circle import Circle
import cv2 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

# Global variables for multiprocessing
target_img_global = None
height_global = None
width_global = None

def init_worker(target_img, height, width):
    """Initialize worker process with shared data"""
    global target_img_global, height_global, width_global
    target_img_global = target_img
    height_global = height
    width_global = width

def mutate(genome, img_width, img_height):
    """Create a mutated genome by adding a random circle"""
    color = (random.random(), random.random(), random.random(), random.random())
    pos = (random.randrange(0, img_width), random.randrange(0, img_height))
    radius = random.randrange(1, 15)  # Avoid radius 0
    return genome + [Circle(pos, radius, color)]

def evaluate_child_fitness(child):
    """Evaluate fitness of a single child - optimized for multiprocessing"""
    global target_img_global, height_global, width_global
    
    # Create image more efficiently
    child_img = np.zeros((height_global, width_global, 3), dtype=np.float32)
    
    # Batch process circles if possible
    for circle in child:
        # Ensure position is within bounds
        pos = (max(0, min(circle.pos[0], width_global-1)), 
               max(0, min(circle.pos[1], height_global-1)))
        radius = max(1, min(circle.radius, min(width_global, height_global)//2))
        
        cv2.circle(child_img, pos, radius, circle.color, thickness=-1)
    
    # Calculate fitness using vectorized operations
    diff = child_img - target_img_global
    fitness = np.mean(diff * diff)  # More efficient than ** 2
    
    return fitness, child

def create_children_batch(args):
    """Create a batch of children for multiprocessing"""
    parent_genome, batch_size, img_width, img_height = args
    return [mutate(parent_genome, img_width, img_height) for _ in range(batch_size)]

def load_image_cv2(filename):
    img = cv2.imread(filename)
    colored_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.array(colored_img, dtype=np.float32) / 255.0  # Normalize to 0-1
    return img_array
  
def show_image_array(img): 
    plt.figure(figsize=(8, 6))
    plt.imshow(np.clip(img, 0, 1))  # Clip values for display
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('current_generation.png', bbox_inches='tight', dpi=100)
    plt.close()

def show_genome(genome, width, height):
    img = np.zeros((height, width, 3), dtype=np.float32)
    for circle in genome:
        pos = (max(0, min(circle.pos[0], width-1)), 
               max(0, min(circle.pos[1], height-1)))
        radius = max(1, min(circle.radius, min(width, height)//2))
        cv2.circle(img, pos, radius, circle.color, thickness=-1)
    show_image_array(img)

def main():
    # Initialize
    parent_genome = []
    target_img = load_image_cv2("target3.png")
    target_img = cv2.resize(target_img, (128, 128))
    target_img = target_img.astype(np.float32) / 255.0  # Normalize

    parent_fit = float('inf')  # Use proper infinity
    height, width = target_img.shape[:2]
    pop_size = 200  # Increased population size for better results
    generation = 0
    
    # Determine number of processes
    num_processes = min(cpu_count(), 4)  # Use up to 4 processes
    batch_size = pop_size // num_processes
    
    print(f"Using {num_processes} processes with batch size {batch_size}")
    
    # Create process pool
    with Pool(processes=num_processes, initializer=init_worker, 
              initargs=(target_img, height, width)) as pool:
        
        while True:
            start_time = time.time()
            
            # Create children in parallel batches
            batch_args = [(parent_genome, batch_size, width, height) 
                         for _ in range(num_processes)]
            
            # Generate children in parallel
            children_batches = pool.map(create_children_batch, batch_args)
            children = [child for batch in children_batches for child in batch]
            
            # Add any remaining children if pop_size doesn't divide evenly
            remaining = pop_size - len(children)
            if remaining > 0:
                children.extend([mutate(parent_genome, width, height) 
                               for _ in range(remaining)])
            
            # Evaluate fitness in parallel
            results = pool.map(evaluate_child_fitness, children)
            
            # Find best child
            best_fitness, best_child = min(results, key=lambda x: x[0])
            
            generation += 1
            
            # Update parent if improvement found
            if best_fitness < parent_fit:
                parent_fit = best_fitness
                parent_genome = best_child
                
                elapsed = time.time() - start_time
                print(f"Generation {generation} - Best Fit: {parent_fit:.6f} - "
                      f"Time: {elapsed:.2f}s - Genome size: {len(parent_genome)}")
            
            # Show progress
            if generation % 50 == 0:  # Show less frequently due to speed
                elapsed = time.time() - start_time
                print(f"Generation {generation} - Current Fit: {parent_fit:.6f} - "
                      f"Time: {elapsed:.2f}s")
                show_genome(parent_genome, width, height)

if __name__ == "__main__":
    main()
