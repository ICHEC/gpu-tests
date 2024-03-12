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
  t1 = time.time() - preMatrix
  print(f"Matrix created in {t1:.3f} s\n")

  preEig = time.time()
  # Calculate eigenvalues and eigenvectors
  eigValues, eigVectors = func.linalg.eigh(matrix)
  t2 = time.time() - preEig
  print(f"Eigen values and vectors calculated in {t2:.3f} s\n")

  preDiagonal = time.time()
  # Diagonalise the original matrix
  diagonalisedMatrix = func.linalg.inv(eigVectors) @ matrix @ eigVectors
  t3 = time.time() - preDiagonal
  print(f"Matrix diagonalized in {t3:.3f} s\n")
  return np.array([t1, t2, t3]), cp.asnumpy(eigValues), cp.asnumpy(eigVectors)

if __name__=="__main__":
  N = 10000
  calculateEigens(size=N, device="CPU")
  calculateEigens(size=N, device="GPU")
  

