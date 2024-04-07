import numpy as np
import random as rd

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

sudoku_og = [
    0, 0, 3, 0, 2, 0, 6, 0, 0,
    9, 0, 0, 3, 0, 5, 0, 0, 1,
    0, 0, 1, 8, 0, 6, 4, 0, 0,
    0, 0, 8, 1, 0, 2, 9, 0, 0,
    7, 0, 0, 0, 0, 0, 0, 0, 8,
    0, 0, 6, 7, 0, 8, 2, 0, 0,
    0, 0, 2, 6, 0, 9, 5, 0, 0,
    8, 0, 0, 2, 0, 3, 0, 0, 9,
    0, 0, 5, 0, 1, 0, 3, 0, 0
]


sudoku_test = [    
    4, 0, 3, 0, 2, 0, 6, 0, 7,
    9, 6, 7, 3, 4, 5, 8, 2, 1,
    2, 5, 1, 8, 7, 6, 4, 9, 3,
    5, 4, 8, 1, 3, 2, 9, 7, 6,
    7, 2, 9, 5, 6, 4, 1, 3, 8,
    1, 3, 6, 7, 9, 8, 2, 4, 5,
    3, 7, 2, 6, 8, 9, 5, 1, 4,
    8, 1, 4, 2, 5, 3, 7, 6, 9,
    6, 9, 5, 4, 1, 7, 3, 8, 2
]

class Particle:
    def __init__(self, grid: np.ndarray, velocity: np.ndarray, bestScore: int):
        self.grid = grid
        self.best = grid
        self.bestScore = bestScore
        self.velocity = velocity

class Swarm:
    def __init__(self, particles: list[Particle], best: np.ndarray, bestScore: int):
        self.particles = particles
        self.best = best
        self.bestScore = bestScore

class Solver:
    def __init__(self, toSolve: np.ndarray, inertiaWeight: float, rhoP: float, rhoG: float, limit: int, nbParticles: int):
        self.toSolve = toSolve
        self.inertiaWeight = inertiaWeight
        self.rhoP = rhoP
        self.rhoG = rhoG
        self.limit = limit
        self.nbParticles = nbParticles
        #self.preFilledIndexes = np.where(toSolve != 0)[0]
        self.swarm = None
    
    # Init the swarm attribute of the solver
    #
    # The swarm is set with nbParticles particles, which contains the preFilledValues, and are filled with random values from 1 to 9
    def initSwarm(self):
        best = self.toSolve
        bestScore = 0
        particles = np.array([])

        for _ in range(self.nbParticles):
            # Fill grid with randoms
            indices = np.where(self.toSolve == 0)
            grid = self.toSolve.copy()
            grid[indices] = np.random.randint(1, 10, size=len(indices[0]))

            # velocity = np.random.randint(17, size=81) - 8
            # for preFilledIndex in self.preFilledIndexes:
            #     velocity[preFilledIndex] = 0

            # Create velocity and set velocity to 0 for pre filled indexes
            velocity = np.zeros(81)

            # Score the grid
            _, gridScore = self.fitness(grid)
            if gridScore > bestScore:
                bestScore = gridScore
                best = grid

            particles = np.append(particles, Particle(grid, velocity, gridScore))

        _, bestScore = self.fitness(best)
        self.swarm = Swarm(particles, best, bestScore)
        
    # Solve the sudoku
    #
    # Solve the sudoku with a main loop which stops when "limit" is reached or when a solution is found
    # It loops over each particles to update their velocity, position, and update the best sudoku in case the new one gets a better score from fitness function
    def solve(self) -> np.ndarray:
        count = 0
        while count < self.limit:
            for particle in self.swarm.particles:
                for i in range(81):
                    r = np.random.uniform(size=2)
                    # update new velocity
                    particle.velocity[i] = (particle.velocity[i] * self.inertiaWeight) + (self.rhoP * r[0] * (particle.best[i] - particle.grid[i])) + (self.rhoG * r[1] * (self.swarm.best[i] - particle.grid[i]))
                
                # update new position
                particle.grid = np.add(particle.grid, particle.velocity)
                particle.grid[particle.grid >= 9.5] = 1
                particle.grid[particle.grid < .5] = 9

                # compare new particle score
                isSolved, particleScore = self.fitness(particle.grid)
                if isSolved:
                    return particle.grid
                if particleScore > particle.bestScore:
                    particle.best = particle.grid
                    particle.bestScore = particleScore
                    if particleScore > self.swarm.bestScore:
                        self.swarm.best = particle.grid
                        self.swarm.bestScore = particleScore
            count += 1
        return self.swarm.best

    # evaluate grid
    # our aim is to maximize the fitness
    def fitness(self, grid: np.ndarray) -> int:
        vfunc = np.vectorize(lambda x: int(round(x)))
        copyGrid = vfunc(grid)

        (squareCount, squareRows, squareColumns) = self.checkSquare(copyGrid)
        (rowSolution, rowsCount, indexRows) = self.checkLine(copyGrid)
        (colSolution, columnsCount, indexColumns) = self.checkColumn(copyGrid)

        fitness = (squareCount * 10) + (rowsCount * 9) + (columnsCount * 9) + (rowsCount * columnsCount)
        
        for i in range(len(indexRows)):
            fitness += squareRows.count(indexRows[i]/3) * 3

        for i in range(len(indexColumns)):
            fitness += squareColumns.count(indexColumns[i]/3) * 3

        return (rowSolution and colSolution, fitness)

    def checkLine(self, grid: np.ndarray, index = -1):
        '''
            grid (np.ndarray) : Array of the sudoku
            index (int) : if between 0-81 -> specific row containing index is checked 
                          else (-1) -> all rows are checked
        '''

        # Check a specific row containing the index
        if index >= 0:
            row = index // 9
            start = row * 9
            end = start + 9
            values = grid[start:end]
            return (True, 1, [row]) if len(set(values)) == len(values) else (False, 0, [])
                

        # Check all rows
        else:
            count = 0
            boolean = True
            l = []
            for row in range(9):
                start = row * 9
                end = start + 9
                values = grid[start:end]
                if len(set(values)) != 9:
                    boolean = False
                else:
                    count += 1
                    l.append(row)
            return (boolean, count, l)

    def checkColumn(self, grid: np.ndarray, index = -1):
        '''
            grid (np.ndarray) : Array of the sudoku
            index (int) : if between 0-81 -> specific column containing index is checked 
                          else (-1) -> all columns are checked
        '''

        # Check a specific column containing the index
        if index >= 0:
            column = index % 9
            values = [grid[i * 9 + column] for i in range(9)]
            return (True, 1, [column] ) if len(set(values)) == len(values) else (False, 0, [])
                

        # Check all columns
        else:
            count = 0
            boolean = True
            l = []
            for column in range(9):
                values = [grid[i * 9 + column] for i in range(9)]
                if len(set(values)) != 9:
                    boolean = False
                else:
                    count += 1
                    l.append(column)
            return (boolean, count, l)
        
    def getSquare(self, x, y, grid):
        startIndex = 27 * y + 3 * x
        square = [grid[startIndex + j + 9 * i] for i in range(3) for j in range(3)]
        return square

    def checkSquare(self, grid):
        rows = [] 
        columns = []
        count = 0
        for y in range(3):
            for x in range(3):
                if len(set(self.getSquare(x, y, grid))) == 9:
                    count += 1
                    rows.append(y)
                    columns.append(x)
        return (count, rows, columns)

if __name__ == '__main__':
    rd.seed()

    solver = Solver(np.array(sudoku_test), inertiaWeight=.75, rhoP=.6, rhoG=.8, nbParticles=100, limit=200)
    solver.initSwarm()
    result = solver.solve()

    print(f"Score: {solver.fitness(result)}")

    vfunc = np.vectorize(lambda x: int(round(x)))
    result = vfunc(result)
    np.savetxt("result", result.reshape((9, 9)), fmt='%d', delimiter='\t')