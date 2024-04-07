import numpy as np
import random as rd
from random import choice

from logzero import logger
# logger.debug(message)
'''
sudoku_solution = [
    4, 8, 3, 9, 2, 1, 6, 5, 7,
    9, 6, 7, 3, 4, 5, 8, 2, 1,
    2, 5, 1, 8, 7, 6, 4, 9, 3,
    5, 4, 8, 1, 3, 2, 9, 7, 6,
    7, 2, 9, 5, 6, 4, 1, 3, 8,
    1, 3, 6, 7, 9, 8, 2, 4, 5,
    3, 7, 2, 6, 8, 9, 5, 1, 4,
    8, 1, 4, 2, 5, 3, 7, 6, 9,
    6, 9, 5, 4, 1, 7, 3, 8, 2
]
'''

sudoku_test = [    
    4, 8, 0, 9, 2, 1, 6, 5, 7,
    9, 6, 7, 3, 4, 5, 8, 2, 0,
    2, 5, 1, 8, 7, 6, 4, 9, 3,
    5, 4, 8, 1, 3, 2, 0, 7, 6,
    7, 0, 9, 0, 6, 4, 1, 3, 8,
    1, 3, 6, 7, 9, 8, 2, 0, 5,
    3, 7, 2, 6, 8, 9, 5, 1, 4,
    8, 1, 0, 2, 5, 3, 7, 6, 0,
    0, 9, 5, 4, 1, 0, 3, 8, 2
]


def compute_error(sudoku: np.ndarray) -> int:
    nb_error = 0
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]

    for i in range(9):
        for j in range(9):
            val = sudoku[i * 9 + j]
            nb_error += (val in rows[i]) + (val in cols[j])
            rows[i].add(val)
            cols[j].add(val)

    return nb_error

class Particle:
    def __init__(self, grid: np.ndarray, velocity: np.ndarray, best_score: int):
        self.grid = grid
        self.best = grid
        self.best_score = best_score
        self.zeros_velocity_count = np.zeros(81)
        self.velocity = velocity

class Swarm:
    def __init__(self, particles: list[Particle], best: np.ndarray, best_score: int):
        self.particles = particles
        self.best = best
        self.best_score = best_score

class Solver:
    def __init__(self, to_solve: np.ndarray, K: float, rhoP: float, rhoG: float, limit: int, nb_particles: int):
        self.to_solve = to_solve
        self.K = K
        self.rhoP = rhoP
        self.rhoG = rhoG
        self.limit = limit
        self.nb_particles = nb_particles
        self.pre_filled_indexes = np.where(to_solve != 0)[0]
        self.empty_indexes = np.where(self.to_solve == 0)
        self.nb_empty_indexes = len(self.empty_indexes[0])
        self.swarms : list[Swarm] = []
        
    def init_particle(self):
        grid = self.to_solve.copy()
        grid[self.empty_indexes] = np.random.randint(1, 10, size=self.nb_empty_indexes)
        return Particle(grid=grid, velocity=np.random.uniform(-2, 2, size=81), best_score=compute_error(grid))
    
    # Init the swarm attribute of the solver
    #
    # The swarm is set with nbParticles particles, which contains the preFilledValues, and are filled with random values from 1 to 9
    def init_swarm(self):
        ps = [self.init_particle() for _ in range(self.nb_particles)]

        for i in range(self.nb_particles):
            swarm_ps = [ps[i-1], ps[i], ps[(i+1) % self.nb_particles]]
            scores = [swarm_ps[0].best_score, swarm_ps[1].best_score, swarm_ps[2].best_score]
            
            swarm = Swarm(particles=swarm_ps, best=swarm_ps[scores.index(min(scores))].best, best_score=min(scores))
            self.swarms.append(swarm)

        
    # Solve the sudoku
    #
    # Solve the sudoku with a main loop which stops when "limit" is reached or when a solution is found
    # It loops over each particles to update their velocity, position, and update the best sudoku in case the new one gets a better score from fitness function
    def solve(self) -> np.ndarray:
        count = 0
        while count < self.limit:
            for swarm in self.swarms:
                for particle in swarm.particles:
                    tmp_velocity = particle.velocity
                    particle.velocity = self.K * (particle.velocity + (self.rhoP * np.random.uniform(size=81) * (particle.best - particle.grid)) + (self.rhoG * np.random.uniform(size=81) * (swarm.best - particle.grid)))
                    particle.velocity[particle.velocity < -2] = -2 
                    particle.velocity[particle.velocity > 2] = 2
                    particle.zeros_velocity_count[particle.velocity == tmp_velocity] += 1
                    particle.zeros_velocity_count[particle.velocity != tmp_velocity] += 0
                    
                    for i in range(81):
                        if i not in self.pre_filled_indexes:
                            if particle.zeros_velocity_count[i] > 500: # this value might change
                                particle.velocity[i] = np.random.uniform(-2, 2)
                                particle.zeros_velocity_count[i] = 0
                                
                            if np.random.uniform() < 2 / (1 + np.exp(-np.abs(particle.velocity[i]))) - 1: # scaling function
                                # Excluding the initial value
                                particle.grid[i] = np.random.choice([j for j in range(1,10) if j != particle.grid[i]])
                                
                    # compare new particle score
                    particle_score = compute_error(particle.grid)
                    if particle_score == 0:
                        return particle.grid
                    if particle_score < particle.best_score:
                        particle.best = particle.grid
                        particle.best_score = particle_score
                        if particle_score < swarm.best_score:
                            swarm.best = particle.grid
                            swarm.best_score = particle_score
            count += 1
            
        return []

if __name__ == '__main__':
    rd.seed()

    solver = Solver(np.array(sudoku_test), K=.7298, rhoP=.25, rhoG=.25, nb_particles=100, limit=5000)
    solver.init_swarm()
    result = solver.solve()
    if len(result) == 0:
        print("No solution found...")
    else:
        np.savetxt("result", result.reshape((9, 9)), fmt='%d', delimiter='\t')