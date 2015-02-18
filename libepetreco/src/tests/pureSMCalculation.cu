/** @file   pureSMCalculation.cu */
/* Author: malte
 *
 * Created on 18. Februar 2015, 14:13 */

#define NBLOCKS 32

#include "wrappers.hpp"
#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "csrmv.hpp"

/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
int const LIMNNZ(134217728);

/* Max # of channels in COO matrix arrays */
int const LIMM(LIMNNZ/VGRIDSIZE);

int main(int argc, char** argv) {
#ifdef MEASURE_TIME
  clock_t time1 = clock();
#endif
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  number of rays" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  
  /* NUMBER OF RAYS PER CHANNEL */
  int const nrays(atoi(argv[2]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  /* MEASUREMENT SETUP */
  MS setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  /* VOXEL GRID */
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  /* MEASUREMENT LIST */
  /* Number of non-zeros, row indices */
  int effM; std::vector<int> yRowId_host;
  
  readMeasList_HDF5<val_t>(yRowId_host, effM, fn);
  
  int * yRowId_devi = NULL;
  HANDLE_ERROR(mallocMeasList_devi(yRowId_devi, effM));
  HANDLE_ERROR(cpyMeasListH2D(yRowId_devi, &(yRowId_host[0]), effM));
  
  /* MAX NUMBER OF NON_ZEROS IN SYSTEM MATRIX */
  int maxNnz = effM * VGRIDSIZE;
  
  /* SYSTEM MATRIX */
  /* Row (channel) ids, row pointers, effective row pointers, column (voxel)
   * ids, values, number of non-zeros (host, devi) */
  int * aCnlId_devi = NULL; int * aVxlId_devi = NULL; val_t * aVal_devi = NULL;
  HANDLE_ERROR(malloc_devi<int>(  aCnlId_devi, LIMM*VGRIDSIZE));
  HANDLE_ERROR(malloc_devi<int>(  aVxlId_devi, LIMM*VGRIDSIZE));
  HANDLE_ERROR(malloc_devi<val_t>(aVal_devi,   LIMM*VGRIDSIZE));
  int * nnz_devi = NULL;
  HANDLE_ERROR(malloc_devi<int>(  nnz_devi,    1));
#ifdef MEASURE_TIME
  clock_t time2 = clock();
  printTimeDiff(time2, time1, "Time before SM calculation: ");
#endif /* MEASURE_TIME */
  
  /* SM CALCULATION */
  for(int chunkId=0; chunkId<nChunks(maxNnz, LIMM*VGRIDSIZE); chunkId++) {
    int m = nInChunk(chunkId, effM, LIMM);
    int ptr = chunkPtr(chunkId, LIMM);
    
    int nnz_host[1] = {0};
    HANDLE_ERROR(memcpyH2D<int>(nnz_devi, nnz_host, 1));
    
    int * m_devi = NULL;
    HANDLE_ERROR(malloc_devi<int>(m_devi, 1));
    HANDLE_ERROR(memcpyH2D<int>(m_devi, &m, 1));
    
    /* Get system matrix */
    getSystemMatrix<
          val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0, Trafo1>
          <<<NBLOCKS, TPB>>>
          ( aVal_devi, aVxlId_devi, aCnlId_devi, &(yRowId_devi[ptr]), m_devi, nnz_devi);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* Cleanup */
    cudaFree(m_devi);
  }
#ifdef MEASURE_TIME
  clock_t time3 = clock();
  printTimeDiff(time3, time2, "Time for SM calculation: ");
#endif /* MEASURE_TIME */
          
  /* Cleanup */
  cudaFree(yRowId_devi);
  cudaFree(aCnlId_devi);
  cudaFree(aVxlId_devi);
  cudaFree(aVal_devi);
  cudaFree(nnz_devi);
#ifdef MEASURE_TIME
  clock_t time4 = clock();
  printTimeDiff(time4, time3, "Time after SM calculation: ");
#endif /* MEASURE_TIME */
  
  return 0;
}

