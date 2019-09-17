import random

class Matrix:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.matrixInit()
    
    # generates matrix with random ints
    def matrixInit(self):
        self.res = [[random.randint(1, 100) for i in range(self.m)] for j in range(self.n)]

    def __add__(self, o):
        if (self.n != o.n or self.m != o.m): return "cannot add, dimensions are different"

        result = Matrix(self.n, self.m)
        for i in range(self.n):
            for j in range(self.m):
                result.res[i][j] = self.res[i][j] + o.res[i][j]
        return result

    def __sub__(self, o):
        if (self.n != o.n or self.m != o.m): return "cannot subtract, dimensions are different"
        
        result = Matrix(self.n, self.m)
        for i in range(self.n):
            for j in range(self.m):
                result.res[i][j] = self.res[i][j] - o.res[i][j]

        return result

    def __mul__(self, o):
        if (self.n != o.m or self.m != o.n): return "cannot multiply, dimensions are different"
        
        result = Matrix(self.n, o.m)
        buffer = 0
        
        for i in range(self.n): 
            for j in range(o.m): 
                for k in range(o.n): 
                    result.res[i][j] += self.res[i][k] * o.res[k][j]

        return result

    def transpose(self):
        buffer = Matrix(self.m, self.n)
        
        for i in range(buffer.n):
            for j in range(buffer.m):
                buffer.res[i][j] = self.res[j][i]

        self.res = buffer.res
        self.n = buffer.n
        self.m = buffer.m

        return self


    def __repr__(self):
        result = [0 for i in range(self.n)]
        for i in range(self.n):
            result[i] = self.res[i]

        return str(result)