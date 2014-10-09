// Example based on:
// http://docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example
// , accessed on 2014-09-30

//Example: Application using C++ and the CUSPARSE library 
//-------------------------------------------------------
#include <iostream>
//#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"

#define CLEANUP \
do { \
    if(cooRowIndex_host)    free(cooRowIndex_host); \
    if(cooColIndex_host)    free(cooColIndex_host); \
    if(cooVal_host)         free(cooVal_host); \
    if(cooRowIndex_devi)    cudaFree(cooRowIndex_devi); \
    if(cooColIndex_devi)    cudaFree(cooColIndex_devi); \
    if(cooVal_devi)         cudaFree(cooVal_devi); \
    if(csrRow_host)         free(csrRow_host); \
    if(csrRow_devi)         cudaFree(csrRow_devi); \
    \
    if(sparseXIndex_host)   free(sparseXIndex_host); \
    if(sparseXVal_host)     free(sparseXVal_host); \
    if(sparseXIndex_devi)   cudaFree(sparseXIndex_devi); \
    if(sparseXVal_devi)     cudaFree(sparseXVal_devi); \
    \
    if(denseY_host)         free(denseY_host); \
    if(denseY_devi)         cudaFree(denseY_devi); \
    \
    if(descr)               cusparseDestroyMatDescr(descr); \
    if(handle)              cusparseDestroy(handle); \
    cudaDeviceReset(); \
    std::cout << std::flush; \
    std::cerr << std::flush; \
} while(false)

void getCudaErrorOutput( cudaError_t err, char const * file, int line ) {
    std::cerr << file << "(" << line << "): cuda error: "
              << cudaGetErrorString(err) << std::endl;
}
void getCusparseErrorOutput( cusparseStatus_t err, char const * file,
                                int line ) {
    std::cerr << file << "(" << line << "): cusparse error"
              << std::endl;
}

#define HANDLE_CUDA_ERROR( err ) \
if(err != cudaSuccess) { \
    getCudaErrorOutput(err, __FILE__, __LINE__); \
    CLEANUP; \
    exit(EXIT_FAILURE); \
}

#define HANDLE_CUSPARSE_ERROR( err ) \
if(err != CUSPARSE_STATUS_SUCCESS) { \
    getCusparseErrorOutput(err, __FILE__, __LINE__); \
    CLEANUP; \
    exit(EXIT_FAILURE); \
}
    


int main()
{
    cusparseHandle_t    handle=0;
    cusparseMatDescr_t  descr=0;
    int *       cooRowIndex_host=0;
    int *       cooColIndex_host=0;
    double *    cooVal_host=0;
    int *       cooRowIndex_devi=0;
    int *       cooColIndex_devi=0;
    double *    cooVal_devi=0;
    int         n;
    int         nnz;
    int *       csrRow_host=0;
    int *       csrRow_devi=0;
    
    int *       sparseXIndex_host=0;
    double *    sparseXVal_host=0;
    int *       sparseXIndex_devi=0;
    double *    sparseXVal_devi=0;
    int         nnzVec;
    
    double *    denseY_host=0;
    double *    denseY_devi=0;
    
    double      dtwo =   2.0;
    double      dthree = 3.0;
    
    /* Create sparse matrix: */
    /* | 1.      2.  3. |
       |     4.         |
       | 5.      6.  7. |
       |     8.      9. | */
    
    n=4; nnz=9;
    
    cooRowIndex_host =  (int*)   malloc(nnz*sizeof(cooRowIndex_host[0]));
    cooColIndex_host =  (int*)   malloc(nnz*sizeof(cooColIndex_host[0]));
    cooVal_host =       (double*)malloc(nnz*sizeof(cooVal_host[0]));
    if((!cooRowIndex_host) || (!cooColIndex_host) || (!cooVal_host)) {
        std::cerr << "Host malloc failed (matrix)" << std::endl;
        CLEANUP;
        exit(EXIT_FAILURE);
    }
    
    cooRowIndex_host[0]=0; cooColIndex_host[0]=0; cooVal_host[0]=1.;
    cooRowIndex_host[1]=0; cooColIndex_host[1]=2; cooVal_host[1]=2.;
    cooRowIndex_host[2]=0; cooColIndex_host[2]=3; cooVal_host[2]=3.;
    cooRowIndex_host[3]=1; cooColIndex_host[3]=1; cooVal_host[3]=4.;
    cooRowIndex_host[4]=2; cooColIndex_host[4]=0; cooVal_host[4]=5.;
    cooRowIndex_host[5]=2; cooColIndex_host[5]=2; cooVal_host[5]=6.;
    cooRowIndex_host[6]=2; cooColIndex_host[6]=3; cooVal_host[6]=7.;
    cooRowIndex_host[7]=3; cooColIndex_host[7]=1; cooVal_host[7]=8.;
    cooRowIndex_host[8]=3; cooColIndex_host[8]=3; cooVal_host[8]=9.;
    
    /* Copy matrix to GPU memory */
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&cooRowIndex_devi,
                       nnz*sizeof(cooRowIndex_devi[0])));
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&cooColIndex_devi,
                       nnz*sizeof(cooColIndex_devi[0])));
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&cooVal_devi,
                       nnz*sizeof(cooVal_devi[0])));
    
    HANDLE_CUDA_ERROR(
            cudaMemcpy(cooRowIndex_devi, cooRowIndex_host,
                       nnz*sizeof(cooRowIndex_host[0]),
                       cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
            cudaMemcpy(cooColIndex_devi, cooColIndex_host,
                       nnz*sizeof(cooColIndex_host[0]),
                       cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
            cudaMemcpy(cooVal_devi, cooVal_host,
                       nnz*sizeof(cooVal_host[0]),
                       cudaMemcpyHostToDevice));
    
    /* Create a dense and a sparse vector: */
    /* y = [ 10., 20., 30., 40., 50., 60., 70., 80. ] 
     * xIndex = [  0    3    5  ]
     * xVal   = [ 10., 20., 30. ]*/
    nnzVec = 3;
    denseY_host =     (double*)malloc(2*n    *sizeof(denseY_host[0]));
    sparseXIndex_host = (int*) malloc(nnzVec *sizeof(sparseXIndex_host[0]));
    sparseXVal_host = (double*)malloc(nnzVec *sizeof(sparseXVal_host[0]));
    if((!denseY_host) || (!sparseXIndex_host) || (!sparseXVal_host)) {
        std::cerr << "Host malloc failed (vectors)" << std::endl;
        CLEANUP;
        exit(EXIT_FAILURE);
    }
    
    denseY_host[0] = 10.; sparseXIndex_host[0] = 0; sparseXVal_host[0] = 100.;
    denseY_host[1] = 20.; sparseXIndex_host[1] = 1; sparseXVal_host[1] = 200.;
    denseY_host[2] = 30.;
    denseY_host[3] = 40.; sparseXIndex_host[2] = 3; sparseXVal_host[2] = 400.;
    denseY_host[4] = 50.;
    denseY_host[5] = 60.;
    denseY_host[6] = 70.;
    denseY_host[7] = 80.;
    
    /* Copy vectors to GPU memory */
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&denseY_devi, 2*n *sizeof(denseY_devi[0])));
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&sparseXIndex_devi,
                       nnzVec *sizeof(sparseXIndex_devi[0])));
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&sparseXVal_devi,
                       nnzVec *sizeof(sparseXVal_devi[0])));
    HANDLE_CUDA_ERROR(
            cudaMemcpy(denseY_devi, denseY_host,
                       2*n *sizeof(denseY_devi[0]), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
            cudaMemcpy(sparseXIndex_devi, sparseXIndex_host,
                       nnzVec *sizeof(sparseXIndex_devi[0]),
                       cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
            cudaMemcpy(sparseXVal_devi, sparseXVal_host,
                       nnzVec *sizeof(sparseXVal_devi[0]),
                       cudaMemcpyHostToDevice));
    
    /* Initialize cusparse library */
    HANDLE_CUSPARSE_ERROR(
            cusparseCreate(&handle));
    
    /* Create and setup matrix descriptor */
    HANDLE_CUSPARSE_ERROR(
            cusparseCreateMatDescr(&descr));
    HANDLE_CUSPARSE_ERROR(
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    HANDLE_CUSPARSE_ERROR(
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    
    /* Convert matrix from COO to CSR format */
    HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&csrRow_devi, (n+1)*sizeof(csrRow_devi[0])));
    HANDLE_CUSPARSE_ERROR(
            cusparseXcoo2csr(handle, cooRowIndex_devi, nnz, n, csrRow_devi,
                             CUSPARSE_INDEX_BASE_ZERO));
    
    /* Scatter vector elements */
    HANDLE_CUSPARSE_ERROR(
            cusparseDsctr(handle, nnzVec, sparseXVal_devi, sparseXIndex_devi,
                          &denseY_devi[n], CUSPARSE_INDEX_BASE_ZERO));
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    
    /* Copy vector to host, print result */
    HANDLE_CUDA_ERROR(
            cudaMemcpy(denseY_host, denseY_devi,
                       2*n *sizeof(denseY_host[0]),
                       cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    std::cout << "Exspected: y = [ 10, 20, 30, 40, 100, 200, 70, 400 ]"
              << std::endl;
    std::cout << "Is:        y = [ ";
    for(int i=0; i<2*n; i++) std::cout << denseY_host[i] << ", ";
    std::cout << std::endl;
    
    /* Matrix vector multiplication */
    HANDLE_CUSPARSE_ERROR(
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           n, n, nnz,
                           &dtwo, descr,
                           cooVal_devi, csrRow_devi, cooColIndex_devi,
                           &denseY_devi[0], &dthree, &denseY_devi[n]));
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    
    /* Copy vector to host, print result */
    HANDLE_CUDA_ERROR(
            cudaMemcpy(denseY_host, denseY_devi,
                       2*n *sizeof(denseY_host[0]),
                       cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    std::cout << "Exspected: y = [ 10, 20, 30, 40, 680, 760, 1230, 2240 ]"
              << std::endl;
    std::cout << "Is:        y = [ ";
    for(int i=0; i<2*n; i++) std::cout << denseY_host[i] << ", ";
    std::cout << std::endl;
    
    CLEANUP;
    return 0;
}
