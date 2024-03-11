import numpy as np
import cupy as cp
import time

def calculateEigens(size = 1000, device="CPU"):
  print(f"Using {device}")
  
  if device=="CPU":
    func = np
  else:
    func = cp

  preMatrix = time.time() 
  matrix = func.random.random([size, size])
  matrix = 0.5 * (matrix + matrix.T) 
  postMatrix = time.time()
  print(f"Matrix created in {(time.time() - preMatrix):.3f} s\n")

  preEig = time.time()
  # Calculate eigenvalues and eigenvectors
  eigValues, eigVectors = func.linalg.eigh(matrix)
  print(f"Eigen values and vectors calculated in {(time.time()-preEig):.3f} s\n")

  preDiagonal = time.time()
  # Diagonalise the original matrix
  diagonalisedMatrix = func.linalg.inv(eigVectors) @ matrix @ eigVectors
  print(f"Matrix diagonalized in {(time.time()-preDiagonal):.3f} s\n")

if __name__=="__main__":
  N = 10000
  calculateEigens(size=N, device="CPU")
  calculateEigens(size=N, device="GPU")
  

