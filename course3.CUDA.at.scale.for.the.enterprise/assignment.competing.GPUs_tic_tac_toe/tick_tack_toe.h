#include <stdio.h>
#include <tuple>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <cuda_runtime.h>

// For the CUDA runtime routines (prefixed with "cuda_")
using namespace std;

__global__ void play(int *board, int runId);
__host__ int firstMove(int idx, int *board);
__host__ int genRandom();
__host__ int firstMove(int idx, int *board);
__device__ std::tuple<int, int, int> checkRow(int idx, int *board, int i);
__device__ std::tuple<int, int, int> checkColumn(int idx, int *board, int j);
__device__ std::tuple<int, int, int> checkLeftDiagonal(int idx, int *board, int i, int j);
__device__ std::tuple<int, int, int> checkRightDiagonal(int idx, int *board, int i, int j);
__device__ int setCellInRow(int idx, int *board, int i);
__device__ int setCellInColumn(int idx, int *board, int j);
__device__ int setCellInLeftDiagnonal(int idx, int *board, int i, int j);
__device__ int setCellInRightDiagnonal(int idx, int *board, int i, int j);
__device__ int getEmptyCellCount(int *board);
__device__ std::tuple<int, int, int, bool> calculateMove(int idx, int rowStats[][3], int colStats[][3], int diagStats[][3]);
__device__ int getChanceFactor(int myCount, int otherCount);