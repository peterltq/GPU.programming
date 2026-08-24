#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <random>

#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

__host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
      // Create a random device to seed the generator
    std::random_device rd;
    
    // Mersenne Twister engine (high quality)
    std::mt19937 gen(rd());
    
    // Define a distribution: integers between 1 and 100 (inclusive)
    std::uniform_int_distribution<int> dist(1, 100);

  // TODO allocate host memory of type float of size height * width called hostMatrix

  // TODO fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue

  float *hostMatrix = (float*) malloc(height * width * sizeof(float));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j)
    hostMatrix[i * HA + j] = (random) ? dist(gen) : nonRandomValue;
  }  

  return hostMatrix;
}

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // TODO allocate device memory of type float of size height * width called deviceMatrix
  long size = height * width * sizeof(float);
  float *deviceMatrix;

  int status = cublasAlloc(height * width, sizeof(float), (void**)&deviceMatrix);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error\n");
      return NULL;
  }

  // TODO set deviceMatrix to values from hostMatrix
  status = cublasSetMatrix(height, width, sizeof(float), hostMatrix, height, deviceMatrix, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device set memory error\n");
      return NULL;
  }
  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  int status;
  status = cublasGetMatrix(height,width,sizeof(float),deviceMatrix,height, hostMemory, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device read error (A)\n");
    return NULL;
  }
  return hostMemory;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cublasStatus status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {

  cublasStatus status;
  cublasInit();

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data
  
  float *A = initializeHostMemory(HA, WA, true, -1);

  float *B = initializeHostMemory(HB, WB, true, -1);

  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    float *C = initializeHostMemory(HC, WC, true, -1);
    
    // TODO create arrays of floats alpha filled with 1's
    float alpha = 1.0f;
    float beta = 0.0f;

    // TODO create arrays of floats beta filled with 0's

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    float *AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
    float *BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
    float *CC = initializeDeviceMemoryFromHostMemory(HC, WC, C);

    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    // This is old version of CUDA (< 6.0).
    cublasSgemm(
       'n', 
       'n',
       HA,WB,WA,
       1,
       AA,HA,
       BB,HB,
       0,
       CC,HC);
    
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }

    C = retrieveDeviceMemory(HC, WC, CC, C);

    printMatrices(A, B, C);

    freeMatrices(A, B, C, AA, BB, CC);
    
    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

}
