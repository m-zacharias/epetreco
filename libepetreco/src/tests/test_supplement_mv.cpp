/** 
 * @file test_supplement_mv.cpp
 */
 /* Author: malte
 *
 * Created on 10. Februar 2015, 13:21
 */

#include "supplement_mv.hpp"

#include <iostream>
#include <cstdlib>
#include <cassert>

#define M 3
#define N 4
#define NRUNS 100

typedef double val_t;

int main(int argc, char** argv) {
  // ###########################################################################
  // ###########################################################################
  // ### Assert
  // ###########################################################################
  // ###########################################################################
  
  for(int run=0; run<NRUNS; run++) {
    // Set dense matrix
    val_t denseMat[M*N];
    int nnz = 0;
    srand(time(NULL));
    for(int i=0; i<M*N; i++) {
      denseMat[i] = rand()%10;
      if(denseMat[i] < 5) {
        denseMat[i] = 0;
      } else {
        nnz++;
      }
    }

    // Convert to COO sparse matrix
    int * matRowId = new int[nnz];
    int * matColId = new int[nnz];
    val_t * matVal = new val_t[nnz];
    int id = 0;
    for(int rowId=0; rowId<M; rowId++) {
      for(int colId=0; colId<N; colId++) {
        if(denseMat[(rowId*N)+colId] != 0) {
          matRowId[id] = rowId;
          matColId[id] = colId;
          matVal[id]   = denseMat[(rowId*N)+colId];
          id++;
        }
      }
    }

    // Convert to CSR sparse matrix
    int matRowPtr[M+1];
    id = 0;
    matRowPtr[0] = 0;
    for(int rowId=0; rowId<M; rowId++) {
      while(matRowId[id] == rowId) {
        id++;
      }
      matRowPtr[rowId+1] = id;
    }
    
    // Set vectors
    val_t denseX1[N];
    val_t denseX2[N];
    val_t denseX[N];
    for(int i=0; i<N; i++) {
      denseX[i] = rand()/100;
      denseX1[i] = denseX[i];
      denseX2[i] = denseX[i];
    }
    val_t denseY1[M];
    val_t denseY2[M];
    val_t denseY[M];
    for(int i=0; i<M; i++) {
      denseY[i] = rand()%100;
      denseY1[i] = denseY[i];
      denseY2[i] = denseY[i];
    }
    
    // Set scalars
    val_t alpha = (val_t)(rand())/RAND_MAX;
    val_t beta  = (val_t)(rand())/RAND_MAX;

    // Matrix vector multiplications
    mv(&alpha, M, N, denseMat, denseX, &beta, denseY1);
    csrMv(&alpha, M, N, nnz, matRowPtr, matColId, matVal, denseX, &beta, denseY2);
    mv_transposed(&alpha, M, N, denseMat, denseY, &beta, denseX1);
    csrMv_transposed(&alpha, M, N, nnz, matRowPtr, matColId, matVal, denseY, &beta, denseX2);
    
    // Assert: Identic results
    for(int i=0; i<M; i++) {
      assert(denseY2[i] == denseY1[i]);
      assert(denseX2[i] == denseX1[i]);
    }
  }
  
//#define DEMO
#ifdef DEMO
  // ###########################################################################
  // ###########################################################################
  // ### DEMO
  // ###########################################################################
  // ###########################################################################
  
  // ###########################################################################
  // ### DENSE MV
  // ###########################################################################
  do {
    std::cout << "DENSE MV:" << std::endl;
    std::cout << "---------" << std::endl;
    std::cout << std::endl;

    // Set dense matrix A
    val_t denseMat[M*N];
    for(int i=0; i<M*N; i++) {
      denseMat[i] = i+1;
    }
    std::cout << "A:" << std::endl;
    for(int i=0; i<M; i++) {
      for(int j=0; j<N; j++) {
        std::cout << denseMat[(i*N)+j] << "  ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // Set vector x
    val_t denseX[N];
    for(int i=0; i<N; i++) {
      denseX[i] = 2;
    }
    std::cout << "x:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseX[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Set vector y
    val_t denseY[M];
    for(int i=0; i<M; i++) {
      denseY[i] = 10.;
    }
    std::cout << "y:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;

    // Set scalars alpha, beta
    val_t alpha = 0.1;
    std::cout << "alpha: " << alpha << std::endl << std::endl;
    val_t beta  = 1.;
    std::cout << "beta: "  << beta  << std::endl << std::endl;

    // Do matrix vector multiplication
    mv(&alpha, M, N, denseMat, denseX, &beta, denseY);

    // Print result
    std::cout << "y after mv:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;
  } while(false);
  
  
  
  // ###########################################################################
  // ### CSR MV
  // ###########################################################################
  do {
    std::cout << "CSR MV:" << std::endl;
    std::cout << "-------" << std::endl;
    std::cout << std::endl;

    // Set number of non-zero elements
    int nnz=M*N;

    // Set matVal
    val_t matVal[M*N];
    for(int i=0; i<(M*N); i++) {
      matVal[i] = (val_t)(i+1);
    }

    // Set matRowPtr
    int matRowPtr[M+1];
    for(int i=0; i<M+1; i++) {
      matRowPtr[i] = i*N;
    }
    std::cout << "matRowPtr:" << std::endl;
    for(int i=0; i<M+1; i++) {
      std::cout << matRowPtr[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Set matColId
    int matColId[M*N];
    for(int i=0; i<M*N; i++) {
      matColId[i] = i%N;
    }
    std::cout << "matColId:" << std::endl;
    for(int i=0; i<(M*N); i++) {
      std::cout << matColId[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Print val
    std::cout << "matVal:" << std::endl;
    for(int i=0; i<M*N; i++) {
      std::cout << matVal[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Set vector x
    val_t denseX[N];
    for(int i=0; i<N; i++) {
      denseX[i] = 2;
    }
    std::cout << "x:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseX[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Set vector y
    val_t denseY[M];
    for(int i=0; i<M; i++) {
      denseY[i] = 10.;
    }
    std::cout << "y:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;

    // Set scalars alpha, beta
    val_t alpha = 0.1;
    std::cout << "alpha: " << alpha << std::endl << std::endl;
    val_t beta  = 1.;
    std::cout << "beta: "  << beta  << std::endl << std::endl;
 
    
    // Do matrix vector multiplication
    csrMv(&alpha, M, N, nnz, matRowPtr, matColId, matVal, denseX, &beta, denseY);

    // Print result
    std::cout << "y after csrMv:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;
  } while(false);



  // ###########################################################################
  // ### DENSE MV TRANSPOSED
  // ###########################################################################
  do {
    std::cout << "DENSE MV TRANSPOSED:" << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << std::endl;

    // Set dense matrix A
    val_t denseMat[M*N];
    for(int i=0; i<M*N; i++) {
      denseMat[i] = i+1;
    }
    std::cout << "A:" << std::endl;
    for(int i=0; i<M; i++) {
      for(int j=0; j<N; j++) {
        std::cout << denseMat[(i*N)+j] << "  ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // Set vector x
    val_t denseX[M];
    for(int i=0; i<M; i++) {
      denseX[i] = 2;
    }
    std::cout << "x:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseX[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Set vector y
    val_t denseY[N];
    for(int i=0; i<N; i++) {
      denseY[i] = 10.;
    }
    std::cout << "y:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;

    // Set scalars alpha, beta
    val_t alpha = 0.1;
    std::cout << "alpha: " << alpha << std::endl << std::endl;
    val_t beta  = 1.;
    std::cout << "beta: "  << beta  << std::endl << std::endl;

    // Do matrix vector multiplication
    mv_transposed(&alpha, M, N, denseMat, denseX, &beta, denseY);

    // Print result
    std::cout << "y after mv:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;
  } while(false);
    
  
  
  // ###########################################################################
  // ### CSR MV TRANSPOSED
  // ###########################################################################
  do {
    std::cout << "CSR MV TRANSPOSED:" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << std::endl;

    // Set number of non-zero elements
    int nnz=M*N;

    // Set matVal
    val_t matVal[M*N];
    for(int i=0; i<(M*N); i++) {
      matVal[i] = (val_t)(i+1);
    }

    // Set matRowPtr
    int matRowPtr[M+1];
    for(int i=0; i<M+1; i++) {
      matRowPtr[i] = i*N;
    }
    std::cout << "matRowPtr:" << std::endl;
    for(int i=0; i<M+1; i++) {
      std::cout << matRowPtr[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Set matColId
    int matColId[M*N];
    for(int i=0; i<M*N; i++) {
      matColId[i] = i%N;
    }
    std::cout << "matColId:" << std::endl;
    for(int i=0; i<(M*N); i++) {
      std::cout << matColId[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Print matVal
    std::cout << "matVal:" << std::endl;
    for(int i=0; i<M*N; i++) {
      std::cout << matVal[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Set vector x
    val_t denseX[M];
    for(int i=0; i<M; i++) {
      denseX[i] = 2;
    }
    std::cout << "x:" << std::endl;
    for(int i=0; i<M; i++) {
      std::cout << denseX[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Set vector y
    val_t denseY[N];
    for(int i=0; i<N; i++) {
      denseY[i] = 10.;
    }
    std::cout << "y:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;

    // Set scalars alpha, beta
    val_t alpha = 0.1;
    std::cout << "alpha: " << alpha << std::endl << std::endl;
    val_t beta  = 1.;
    std::cout << "beta: "  << beta  << std::endl << std::endl;

    // Do matrix vector multiplication
    csrMv_transposed(&alpha, M, N, nnz, matRowPtr, matColId, matVal, denseX, &beta, denseY);

    // Print result
    std::cout << "y after csrMv:" << std::endl;
    for(int i=0; i<N; i++) {
      std::cout << denseY[i] << std::endl;
    }
    std::cout << std::endl;
  } while(false);
#endif
  
  
  
  return 0;
}

