import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=1.0, #0.5
            emptyPercentage=-0.6, #0.6
            linearity=1.0, #-0.5
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        left = 1
        right = width - 1

        if random.random() < 0.5:
            for y in range(height - 1):
                for x in range(left, right):
                    if (genome[y][x] == 'T' or genome[y][x] == '|') and genome[y-1][x] == "-":
                            genome[y][x] = "-"
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                if y > 0:
                    new_genome[y][x] = random.choice([self.genome[y][x], other.genome[y][x]])
        self.mutate(new_genome)
        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=1.0, #0.5
            emptyPercentage=-0.6, #0.6
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        stair_count = len(list(filter(lambda de: de[1] == "6_stairs", self.genome)))
        penalties -= max(0, (stair_count - 5) * 0.5) 

        #add more enemies to a level
        enemy_count = len(list(filter(lambda de: de[1] == "2_enemy", self.genome)))
        if(enemy_count < 20):
            penalties += .5 * enemy_count
        
        #avoid pipes that are too high for the player to jump over
        max_pipe_height = 4
        tall_pipe_count = len(list(filter(lambda de: de[1] == "7_pipe" and de[2] > max_pipe_height, self.genome)))
        penalties -= tall_pipe_count * 5  

        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.3:  # 30% mutation chance
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            
            x = de[0]
            de_type = de[1]
            choice = random.random()
            
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.5:
                    x = x + random.choice([-1, 1])
                else:
                    breakable = not breakable
                new_de = (x, de_type, y, breakable)
                
            elif de_type == "2_enemy":
                if choice < 0.5:
                    x = x + random.choice([-1, 1])
                new_de = (x, de_type)
                
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = x + random.choice([-1, 1])
                else:
                    h = max(2, min(h + random.choice([-1, 1]), 4))  # Keep height in a good range
                new_de = (x, de_type, h)
            
            new_genome[to_change] = new_de
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        if not self.genome or not other.genome:
            return Individual_DE([]), Individual_DE([])


        pa = random.randint(0, len(self.genome) - 1) if len(self.genome) > 1 else 0
        pb = random.randint(0, len(other.genome) - 1) if len(other.genome) > 1 else 0

        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part

        if not ga:
            ga = self.genome[:]
        if not gb:
            gb = other.genome[:]

        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_Grid  # or Individual_DE


def generate_successors(population):
    results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.

    number_to_select = int(len(population)/2); 

    elitist = elitist_selection(population, number_to_select)
    roulette = roulette_selection(population, number_to_select)

    valid_elitist = [ind for ind in elitist if ind.genome]
    valid_roulette = [ind for ind in roulette if ind.genome]

    if not valid_elitist or not valid_roulette:
        return population

    #generate children with the first result from elitist and roulette
    #elitist[0].generate_children(roulette[0])
    results.extend(elitist[0].generate_children(roulette[0]))

    while len(results) < len(population):
        parent1 = random.choice(elitist)
        parent2 = random.choice(roulette)

        #skip if either parent is empty
        if parent1.genome == [] or parent2.genome == []:
            continue

        children = parent1.generate_children(parent2)
        #results += children
        if children:
            results.extend(children)

    #print("Genome sizes of new generation:", [len(ind.genome) for ind in results])

    return results


def elitist_selection(population, number_to_select):
    #selection method 1: elitist selection
    #Returns the top performing members of the population
    sorted_population = sorted(population, key=lambda ind: (ind.fitness(), len(ind.genome)), reverse=True)
    return sorted_population[:number_to_select]

def roulette_selection(population, number_to_select):
    #selection method 2: roulette selection
    #Returns a random selection of the population, assigning a weight to each individual based on their fitness
    # Ensure no division by zero by adding a small value if fitness is zero
    fitness_arr = [max(0.01, ind.fitness() + len(ind.genome)) for ind in population]  # Ensure fitness is always >0
    total_fitness = sum(fitness_arr)
    probabilities = [f / total_fitness for f in fitness_arr]
    return random.choices(population, probabilities, k=number_to_select)

def ga():
    pop_limit = 480  # Population size
    batches = os.cpu_count()
    batch_size = max(1, pop_limit // (2 * batches))  # Reduce batch size for faster feedback

    max_generations = 1000
    stagnation_limit = 50  # Stop if best fitness hasn't improved for 10 gens
    best_fitnesses = []

    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        
        # Generate initial population (90% random, 10% empty)
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _ in range(pop_limit)]
        
        # Calculate fitness in parallel
        population = pool.map(Individual.calculate_fitness, population, batch_size)
        
        init_done = time.time()
        print(f"Created and calculated initial population in: {init_done - init_time:.2f} seconds")
        
        generation = 0
        start = time.time()

        try:
            while True:
                now = time.time()
                
                # Print statistics
                fitness_scores = [ind.fitness() for ind in population]
                max_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)

                print(f"\nGeneration {generation}")
                print(f"  Max Fitness: {max_fitness}")
                print(f"  Avg Fitness: {avg_fitness:.2f}")
                print(f"  Generation Time: {(now - start) / (generation + 1):.2f} sec")
                print(f"  Net Run Time: {now - start:.2f} sec")

                # Save the best level
                best = max(population, key=Individual.fitness)
                with open("levels/last.txt", 'w') as f:
                    for row in best.to_level():
                        f.write("".join(row) + "\n")
                
                # Check stopping conditions
                best_fitnesses.append(max_fitness)
                if generation >= max_generations:
                    print("Stopping: Max generations reached.")
                    break

                if len(best_fitnesses) >= stagnation_limit and max(best_fitnesses[-stagnation_limit:]) == best_fitnesses[-1]:
                    print("Stopping: No improvement in last", stagnation_limit, "generations.")
                    break
                
                # Generate next generation
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print(f"  Generated successors in: {gendone - gentime:.2f} sec")

                # Recalculate fitness in parallel
                next_population = pool.map(Individual.calculate_fitness, next_population, batch_size)
                popdone = time.time()
                print(f"  Recalculated fitness in: {popdone - gendone:.2f} sec")

                population = next_population
                generation += 1

        except KeyboardInterrupt:
            print("\nProcess manually terminated.")

    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print(f"Best Fitness: {best.fitness()}")

    now = time.strftime("%m_%d_%H_%M_%S")
    for k in range(10):
        with open(f"levels/{now}_{k}.txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")

