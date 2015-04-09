/** @file reco.cu */
/* Author: malte
 *
 * Created on 6. Februar 2015, 17:09 */

#define NBLOCKS 32

#include "wrappers.hpp"
#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "measure_time.hpp"
#include "typedefs.hpp"
#include "csrmv.hpp"
#include "mlemOperations.hpp"

#include <cusparse.h>
#include <sstream>
#include <cstdlib>

/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
MemArrSizeType const LIMNNZ(134217728);

/* Max # of channels in COO matrix arrays */
ListSizeType const LIMM(LIMNNZ/VGRIDSIZE);

int main(int argc, char** argv) {
#if MEASURE_TIME
  clock_t time1 = clock();
#endif /* MEASURE_TIME */
  int const nargs(3);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  filename of output" << std::endl
              << "  number of rays" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  std::string const on(argv[2]);
  
  /* NUMBER OF RAYS PER CHANNEL */
  int const nrays(atoi(argv[3]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  /* MEASUREMENT SETUP */
  MS setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  /* VOXEL GRID */
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  
  
  /* MEASUREMENT VECTOR Y */
  /* Number of non-zeros, row indices, values. */
  ListSizeType effM; std::vector<int> yRowId_host; std::vector<val_t> yVal_host;
  
  do {
    int tmp_effM(0);
    readMeasVct_HDF5(yRowId_host, yVal_host, tmp_effM, fn);
    effM = ListSizeType(tmp_effM);
  } while(false);

  int * yRowId_devi = NULL;
  val_t * yVal_devi = NULL;
  HANDLE_ERROR(mallocSparseVct_devi(yRowId_devi, yVal_devi, effM));
  HANDLE_ERROR(cpySparseVctH2D(yRowId_devi, yVal_devi, &yRowId_host[0], &yVal_host[0], effM));
  
  /* SIMULATED MEASUREMENT VECTOR */
  val_t * yTildeVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(yTildeVal_devi, LIMM));
  
  /* "ERROR" */
  val_t * eVal_devi = NULL; 
  HANDLE_ERROR(malloc_devi(eVal_devi, LIMM));
  
  
  
  /* STUFF FOR MV */
  cusparseHandle_t handle = NULL; cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  val_t zero = val_t(0.); val_t one = val_t(1.);
  
  /* MAX NUMBER OF NON_ZEROS IN SYSTEM MATRIX */
  MemArrSizeType maxNnz(MemArrSizeType(effM) * MemArrSizeType(VGRIDSIZE));
  
  
  
  /* DENSITY X */
  val_t x_host[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) { x_host[i] = 1.; }
  val_t * x_devi = NULL;
  HANDLE_ERROR(malloc_devi<val_t>(x_devi, VGRIDSIZE));
  HANDLE_ERROR(memcpyH2D<val_t>(x_devi, x_host, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  do {
    val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
    scales<val_t>(x_devi, (1./norm), VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
  } while(false);
  
  /* INTERMEDIATE DENSITY GUESS */
//  val_t xx_host[VGRIDSIZE];
  val_t * xx_devi = NULL;
  HANDLE_ERROR(malloc_devi(xx_devi, VGRIDSIZE));
  
  /* CORRECTION */
  val_t c_host[VGRIDSIZE];
  val_t * c_devi = NULL;
  HANDLE_ERROR(malloc_devi(c_devi, VGRIDSIZE));
  
  /* SENSITIVITY */
  val_t s_host[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) { s_host[i]=1; }
  val_t * s_devi = NULL;
  HANDLE_ERROR(malloc_devi(s_devi, VGRIDSIZE));
  HANDLE_ERROR(memcpyH2D<val_t>(s_devi, s_host, VGRIDSIZE));
  
  
  
  /* SYSTEM MATRIX */
  /* Row (channel) ids, row pointers, effective row pointers, column (voxel)
   * ids, values, number of non-zeros (host, devi) */
  int * aCnlId_devi = NULL; int * aCsrCnlPtr_devi = NULL;
  int * aEcsrCnlPtr_devi = NULL; int * aVxlId_devi = NULL;
  val_t * aVal_devi = NULL;
  HANDLE_ERROR(mallocSystemMatrix_devi<val_t>(aCnlId_devi, aCsrCnlPtr_devi,
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi, NCHANNELS, LIMM, VGRIDSIZE));
  MemArrSizeType * nnz_devi = NULL;
  HANDLE_ERROR(malloc_devi<MemArrSizeType>(nnz_devi,          1));
#if MEASURE_TIME
  clock_t time2 = clock();
  printTimeDiff(time2, time1, "Time before BP: ");
#endif /* MEASURE_TIME */

  for(int it=0; it<5; it++) {
    /* Correction to zero */
    for(int i=0; i<VGRIDSIZE; i++) { c_host[i]=0; };
    HANDLE_ERROR(memcpyH2D<val_t>(c_devi, c_host, VGRIDSIZE));
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* CHUNKWISE */
    ChunkGridSizeType NChunks(nChunks<ChunkGridSizeType, MemArrSizeType>
          (maxNnz, MemArrSizeType(LIMM*VGRIDSIZE))
    );
    for(ChunkGridSizeType chunkId=0;
          chunkId<NChunks;
          chunkId++) {
      ListSizeType m   = nInChunk(chunkId, effM, LIMM);
      ListSizeType ptr = chunkPtr(chunkId, LIMM);

      MemArrSizeType nnz_host[1] = {0};
      HANDLE_ERROR(memcpyH2D<MemArrSizeType>(nnz_devi, nnz_host, 1));

      /* Get system matrix */
      systemMatrixCalculation<val_t> (
            aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
            nnz_devi,
            aCnlId_devi, aCsrCnlPtr_devi,
            &(yRowId_devi[ptr]), &m,
            handle);
      HANDLE_ERROR(cudaDeviceSynchronize());
      HANDLE_ERROR(memcpyD2H<MemArrSizeType>(nnz_host, nnz_devi, 1));
      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Simulate measurement */
      CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            m, VGRIDSIZE, *nnz_host, &one, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
            x_devi, &zero, yTildeVal_devi);
      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Calculate "error" */
      divides<val_t>(eVal_devi, &(yVal_devi[ptr]), yTildeVal_devi,
            m);
      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Backproject error */
      CSRmv<val_t>()(handle, CUSPARSE_OPERATION_TRANSPOSE,
            m, VGRIDSIZE, *nnz_host, &one, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
            eVal_devi, &one, c_devi);
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
    
    /* Improve guess */
    dividesMultiplies<val_t>(xx_devi, x_devi, c_devi, s_devi, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* Copy */
    HANDLE_ERROR(memcpyD2D(x_devi, xx_devi, VGRIDSIZE));
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* Normalize */
    val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
    scales<val_t>(x_devi, val_t(1./norm), VGRIDSIZE);
    
    /* Write to file */
    HANDLE_ERROR(memcpyD2H<val_t>(x_host, x_devi, VGRIDSIZE));
    HANDLE_ERROR(cudaDeviceSynchronize());
    std::stringstream ss("");
    ss << it;
    writeDensity_HDF5(x_host, ss.str() + std::string("_") + on, grid);
  }
#if MEASURE_TIME
  clock_t time3 = clock();
  printTimeDiff(time3, time2, "Time for BP: ");
#endif /* MEASURE_TIME */
  
  
  /* Cleanup */
  cudaFree(yRowId_devi);
  cudaFree(yVal_devi);
  cusparseDestroy(handle);
  cusparseDestroyMatDescr(A);
  cudaFree(x_devi);
  cudaFree(aCnlId_devi);
  cudaFree(aCsrCnlPtr_devi);
  cudaFree(aEcsrCnlPtr_devi);
  cudaFree(aVxlId_devi);
  cudaFree(aVal_devi);
  cudaFree(nnz_devi);
#if MEASURE_TIME
  clock_t time4 = clock();
  printTimeDiff(time4, time3, "Time after BP: ");
#endif /* MEASURE_TIME */
  
  return 0;
}

