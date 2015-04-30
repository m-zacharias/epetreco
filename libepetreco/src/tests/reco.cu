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
#include <fstream>
#include <mpi.h>

/* [512 * 1024 * 1024 / 4] (512 MiB of float or int); max # of elems in COO
 * matrix arrays on GPU */
MemArrSizeType const LIMBYTES(512*1024*1024);
MemArrSizeType const LIMNNZ(LIMBYTES/MemArrSizeType(sizeof(val_t)));

/* Max # of channels in COO matrix arrays */
ListSizeType const LIMM(LIMNNZ/VGRIDSIZE);

int main(int argc, char** argv) {
  
#if MEASURE_TIME
  clock_t time1 = clock();
#endif /* MEASURE_TIME */

  int mpi_rank;
  int mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  int const nargs(6);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  filename of output" << std::endl
              << "  number of rays" << std::endl
              << "  filename of sensitivity" << std::endl
              << "  number reco iterations" << std::endl
              << "  filename of density guess" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  std::string const on(argv[2]);
  std::string const sfn(argv[4]);
  int const nIt(atoi(argv[5]));
  std::string const xfn(argv[6]);
  
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
  mallocSparseVct_devi(yRowId_devi, yVal_devi, effM);
  cpySparseVctH2D(yRowId_devi, yVal_devi, &yRowId_host[0], &yVal_host[0], effM);
  
  /* SIMULATED MEASUREMENT VECTOR */
  val_t * yTildeVal_devi = NULL;
  malloc_devi(yTildeVal_devi, LIMM);
  
  /* "ERROR" */
  val_t * eVal_devi = NULL; 
  malloc_devi(eVal_devi, LIMM);
  
  
  
  /* STUFF FOR MV */
  cusparseHandle_t handle = NULL; cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  val_t zero = val_t(0.); val_t one = val_t(1.);
  
  /* MAX NUMBER OF NON_ZEROS IN SYSTEM MATRIX */
  MemArrSizeType maxNnz(MemArrSizeType(effM) * MemArrSizeType(VGRIDSIZE));
  
  
  
  /* DENSITY X */
  bool xfile_good(false);
  do {
    std::ifstream xfile(xfn.c_str());
    xfile_good = xfile.is_open();
  } while(false);
  
  val_t * x_host = NULL;
  if(xfile_good) {
    std::cout << "Will use density from file " << xfn << std::endl;
    readDensity_HDF5(x_host, xfn);
  } else {
    std::cout << "No valid density input file given. Will use homogenous density." << std::endl;
    x_host = new val_t[VGRIDSIZE];
    for(int i=0; i<VGRIDSIZE; i++) { x_host[i] = 1.; }
  }
  
  val_t * x_devi = NULL;
  malloc_devi<val_t>(x_devi, VGRIDSIZE);
  memcpyH2D<val_t>(x_devi, x_host, VGRIDSIZE);
  HANDLE_ERROR(cudaDeviceSynchronize());
  if(!xfile_good) {
    do {
      val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());
      scales<val_t>(x_devi, (1./norm), VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());
    } while(false);
  }
  
  /* INTERMEDIATE DENSITY GUESS */
//  val_t xx_host[VGRIDSIZE];
  val_t * xx_devi = NULL;
  malloc_devi(xx_devi, VGRIDSIZE);
  
  /* CORRECTION */
  val_t c_host[VGRIDSIZE];
  val_t cMpi[VGRIDSIZE];
  val_t * c_devi = NULL;
  malloc_devi(c_devi, VGRIDSIZE);
  
  /* SENSITIVITY */
  val_t s_host[VGRIDSIZE];
  readDensity_HDF5(s_host, sfn);
//  for(int i=0; i<VGRIDSIZE; i++) { s_host[i]=1; }

  val_t * s_devi = NULL;
  malloc_devi(s_devi, VGRIDSIZE);
  memcpyH2D<val_t>(s_devi, s_host, VGRIDSIZE);
  
  /* Normalize */
  val_t norm = sum<val_t>(s_devi, VGRIDSIZE);
  scales<val_t>(s_devi, val_t(1./norm), VGRIDSIZE);
  
  
  
  /* SYSTEM MATRIX */
  /* Row (channel) ids, row pointers, effective row pointers, column (voxel)
   * ids, values, number of non-zeros (host, devi) */
  int * aCnlId_devi = NULL; int * aCsrCnlPtr_devi = NULL;
  int * aEcsrCnlPtr_devi = NULL; int * aVxlId_devi = NULL;
  val_t * aVal_devi = NULL;
  mallocSystemMatrix_devi<val_t>(aCnlId_devi, aCsrCnlPtr_devi,
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi, NCHANNELS, LIMM, VGRIDSIZE);
  MemArrSizeType * nnz_devi = NULL;
  malloc_devi<MemArrSizeType>(nnz_devi,          1);
  
#if MEASURE_TIME
  clock_t * itTimes = new clock_t[nIt+1];
  itTimes[0] = clock();
  if(mpi_rank == 0)
    printTimeDiff(itTimes[0], time1, "Time before reco iterations: ");
#endif /* MEASURE_TIME */

  /* How many chunks are needed? */
  ChunkGridSizeType NChunks(nChunks<ChunkGridSizeType, MemArrSizeType>(maxNnz, MemArrSizeType(LIMM*VGRIDSIZE)));
  
  /* RECO ITERATIONS */
  for(int it=0; it<nIt; it++) {
    
    /* Correction to zero */
    for(int i=0; i<VGRIDSIZE; i++) { c_host[i]=0; };
    memcpyH2D<val_t>(c_devi, c_host, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* CHUNKWISE */
    ChunkGridSizeType chunkId = ChunkGridSizeType(mpi_rank);
    while(chunkId < NChunks) {
      ListSizeType m   = nInChunk(chunkId, effM, LIMM);
      ListSizeType ptr = chunkPtr(chunkId, LIMM);

      MemArrSizeType nnz_host[1] = {0};
      memcpyH2D<MemArrSizeType>(nnz_devi, nnz_host, 1);

      /* Get system matrix */
      systemMatrixCalculation<val_t> (
            aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
            nnz_devi,
            aCnlId_devi, aCsrCnlPtr_devi,
            &(yRowId_devi[ptr]), &m,
            handle);
      HANDLE_ERROR(cudaDeviceSynchronize());
      memcpyD2H<MemArrSizeType>(nnz_host, nnz_devi, 1);
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
      
      /* Go for next chunk */
      chunkId += mpi_size;
    } /* while(chunkId < NChunks) */
    
    /* Copy to host and reduce correction between prodcesses */
    memcpyD2H(c_host, c_devi, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
    MPI_Reduce(c_host, cMpi, VGRIDSIZE, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    /* Change and save density guess */
    if(mpi_rank == 0) {
      memcpyH2D(c_devi, cMpi, VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());
      
      /* Improve guess */
      dividesMultiplies<val_t>(xx_devi, x_devi, c_devi, s_devi, VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Copy */
      memcpyD2D(x_devi, xx_devi, VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Normalize */
      val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
      scales<val_t>(x_devi, val_t(1./norm), VGRIDSIZE);

      /* Write to file */
      memcpyD2H<val_t>(x_host, x_devi, VGRIDSIZE);
      HANDLE_ERROR(cudaDeviceSynchronize());
      std::stringstream ss("");
      ss << it;
      writeDensity_HDF5(x_host, ss.str() + std::string("_") + on, grid);
      
#if MEASURE_TIME
      itTimes[it+1] = clock();
      if(mpi_rank == 0)
        printTimeDiff(itTimes[it+1], itTimes[it], "Time for latest reco iteration: ");
#endif
      
    } /* if(mpi_rank == 0) */
    
  } /* for(int it=0; it<nIt; it++) */

#if MEASURE_TIME
  clock_t time3 = clock();
  if(mpi_rank == 0)
    printTimeDiff(time3, itTimes[0], "Time for reco iterations: ");
  delete[] itTimes;
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
  
  MPI_Finalize();
  
#if MEASURE_TIME
  clock_t time4 = clock();
  if(mpi_rank == 0)
    printTimeDiff(time4, time3, "Time after reco iterations: ");
#endif /* MEASURE_TIME */
  
  return 0;
}

