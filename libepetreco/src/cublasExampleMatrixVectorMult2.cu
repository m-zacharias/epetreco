#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#define \
CHALLENGE( b, m ) \
{ if((b)) { std::cerr << (m) << std::endl; exit(EXIT_FAILURE); } }

// leaDim is the leading dimension - for cublas this is the number elements in
// one column.
#define LINID( rowId, colId, leaDim ) (((colId)*(leaDim))+(rowId))

#define NCOLS 4
#define NROWS 6
#define COLDIM NROWS
#define ROWDIM NCOLS

int main()
{
  typedef float val_t;

  cudaError_t     cudaStat;
  cublasStatus_t  cublasStat;
  cublasHandle_t  cublasHandle;

  val_t * A_host = 0, * x_host = 0, * y_host = 0;
  val_t * A_devi,     * x_devi,     * y_devi;

  val_t alpha = 1., beta = 0.;

  /* Host memory allocation, initialization */
  A_host = (val_t *) malloc(NROWS*NCOLS*sizeof(*A_host));
  CHALLENGE(!A_host, "Host memory allocation failed");
  /*
   * / 1, 1, 1, 1 \
   * | 2, 2, 2, 2 |
   * | 3, 3, 3, 3 |
   * | 4, 4, 4, 4 |
   * | 5, 5, 5, 5 |
   * \ 6, 6, 6, 6 /
   */
  for(int colId=0; colId<NCOLS; colId++)
    for(int rowId=0; rowId<NROWS; rowId++)
      A_host[LINID(rowId, colId, COLDIM)] = rowId+1;

  x_host = (val_t *) malloc(NCOLS*sizeof(val_t));
  CHALLENGE(!x_host, "Host memory allocation failed");
  /*
   * ( 1, 1, 1, 1 )
   */
  for(int colId=0; colId<NCOLS; colId++)
    x_host[colId] = 1;

  y_host = (val_t *) malloc(NROWS*sizeof(val_t));
  CHALLENGE(!y_host, "Host memory allocation failed");
  /*
   * / 0 \
   * | 0 |
   * | 0 |
   * | 0 |
   * | 0 |
   * \ 0 /
   */
  for(int rowId=0; rowId<NROWS; rowId++)
    y_host[rowId] = 0;

  /* Device memory allocation */
  cudaStat = cudaMalloc((void**)&A_devi, NROWS*NCOLS*sizeof(val_t));
  CHALLENGE(cudaStat!=cudaSuccess, "Device memory allocation failed");

  cudaStat = cudaMalloc((void**)&x_devi, NCOLS*sizeof(val_t));
  CHALLENGE(cudaStat!=cudaSuccess, "Device memory allocation failed");

  cudaStat = cudaMalloc((void**)&y_devi, NROWS*sizeof(val_t));
  CHALLENGE(cudaStat!=cudaSuccess, "Device memory allocation failed");

  /* CUBLAS initialization */
  cublasStat = cublasCreate(&cublasHandle);
  CHALLENGE(cublasStat!=CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed");

  /* Data download */
  cublasStat = cublasSetMatrix(
        NROWS, NCOLS, sizeof(val_t), A_host, NROWS, A_devi, NROWS);
  CHALLENGE(cublasStat!=CUBLAS_STATUS_SUCCESS, "Matrix download failed");

  cublasStat = cublasSetVector(NCOLS, sizeof(val_t), x_host, 1, x_devi, 1);
  CHALLENGE(cublasStat!=CUBLAS_STATUS_SUCCESS, "Vector x download failed");
 
  /* Matrix Vector Multiplication */
  cublasStat = cublasSgemv(
        cublasHandle, CUBLAS_OP_N, NROWS, NCOLS, &alpha, A_devi, NROWS, x_devi,
        1, &beta, y_devi, 1);
  CHALLENGE(
        cublasStat!=CUBLAS_STATUS_SUCCESS,
        "Matrix Vector Multiplication failed");

  /* Data upload */
  cublasStat = cublasGetVector(NROWS, sizeof(val_t), y_devi, 1, y_host, 1);
  CHALLENGE(cublasStat!=CUBLAS_STATUS_SUCCESS, "Data upload failed");

  /* Output */
  for(int rowId=0; rowId<NROWS; rowId++)
    std::cout << "y[" << rowId << "] = " << y_host[rowId] << std::endl;

  /* Release resources */
  cudaFree(A_devi);
  cudaFree(x_devi);
  cudaFree(y_devi);
  cublasDestroy(cublasHandle);

  free(A_host);
  free(x_host);
  free(y_host);

  exit(EXIT_SUCCESS);
}
