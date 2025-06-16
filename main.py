import numpy as np
import random
from Circle import Circle
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time


POP_SIZE = 1000
MAX_GENOME_LENGTH = 200
GENE_DELETION_CHANCE = 0.1
IMG_PATH = "target3.png"


def load_image_cv2(filename):
    img = cv2.imread(filename)
    colored_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(colored_img, dtype=np.float32) / 255.0  # Normalize for blending

def show_image_array(img):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('current_generation.png', bbox_inches='tight', dpi=100)
    plt.close()
    from IPython.display import Image, display
    display(Image('current_generation.png'))

def show_genome(genome, width, height):
    img = render_genome(genome, width, height)
    show_image_array(img)

def render_genome(genome, width, height):
    img = np.zeros((height, width, 3), dtype=np.float32)
    for circle in genome:
        cv2.circle(img, circle.pos, circle.radius, circle.color, thickness=-1)
    return img

# ----- GENETIC OPS -----
def mutate(genome, img_width, img_height):
    new_genome = genome.copy()

    # Optional deletion mutation
    if len(new_genome) > 5 and random.random() < GENE_DELETION_CHANCE:
        del new_genome[random.randint(0, len(new_genome) - 1)]

    # Add a new random circle
    color = (random.random(), random.random(), random.random(), random.random())
    pos = (random.randint(0, img_width-1), random.randint(0, img_height-1))
    radius = random.randint(5, img_width // 4)
    new_genome.append(Circle(pos, radius, color))

    # Enforce max length
    if len(new_genome) > MAX_GENOME_LENGTH:
        new_genome = new_genome[-MAX_GENOME_LENGTH:]

    return new_genome

def evaluate_child(args):
    parent_genome, width, height, target_img = args
    child = mutate(parent_genome, width, height)
    child_img = render_genome(child, width, height)
    child_fit = np.mean((child_img - target_img) ** 2)
    return (child_fit, child)

# ----- MAIN EVOLUTION LOOP -----
def evolve():
    parent_genome = []
    target_img = load_image_cv2(IMG_PATH)
    height, width = target_img.shape[:2]
    parent_fit = float('inf')
    generation = 0

    while True:
        # Generate args for parallel children
        args = [(parent_genome, width, height, target_img) for _ in range(POP_SIZE)]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(evaluate_child, args)

        best_child_fit, best_child = min(results, key=lambda x: x[0])

        if best_child_fit < parent_fit:
            parent_genome = best_child
            parent_fit = best_child_fit

        generation += 1
        print(f"Generation {generation} - Best Fit: {parent_fit:.6f} - Genome Length: {len(parent_genome)}")

        if generation % 10 == 0 or generation == 1:
            show_genome(parent_genome, width, height)
            time.sleep(0.5)

if __name__ == "__main__":
    evolve()
