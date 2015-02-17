/** @file reco.cu */
/* Author: malte
 *
 * Created on 6. Februar 2015, 17:09 */

#include <cstdlib>

#define NBLOCKS 32

#include "wrappers.hpp"

#include "CUDA_HandleError.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "typedefs.hpp"
#include "device_constant_memory.hpp"
#include "voxelgrid64_defines.h"
#include "real_measurementsetup_defines.h"
#include "getSystemMatrixDeviceOnly.cu"
#include <cusparse.h>
#include "csrmv.hpp"
#include "mlemOperations.hpp"



int main(int argc, char** argv) {
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
  
  
   /* MEASUREMENT VECTOR */

  /* Number of non-zero elements in measurement vector y. */
  int effM_host[1];
  
  /* Measurement vector row indices. Part of sparse representation. */
  int * yRowId_host = NULL;
  
  /* Measurement vector values. Part of sparse representation. */
  val_t * yVal_host = NULL;
  
  readMeasVct_HDF5(yRowId_host, yVal_host, effM_host[0], fn);
  
  int * effM_devi = NULL;
  int * yRowId_devi = NULL;
  val_t * yVal_devi = NULL;
  HANDLE_ERROR(mallocSparseVct_devi(yRowId_devi, yVal_devi, effM_host[0]));
  HANDLE_ERROR(cpySparseVctH2D(yRowId_devi, yVal_devi, yRowId_host, yVal_host,
        effM_host[0]));
  
  
  /* MAX NUMBER OF NON_ZEROS IN SYSTEM MATRIX */
  int maxNnz = effM_host[0] * VGRIDSIZE;
  
  
  /* SYSTEM MATRIX */
  
  /* System matrix row ids (which are channel ids). Part of COO representation. */
  int * aCnlId_devi = NULL;
  HANDLE_ERROR(malloc_devi(aCnlId_devi, maxNnz));
  
  /* System matrix row pointers. Part of CSR representation. */
  int * aCsrCnlPtr_devi = NULL;
  HANDLE_ERROR(malloc_devi(aCsrCnlPtr_devi, NCHANNELS+1));
  
  /* System matrix effective row pointers. Part of ECSR representation. */
  int * aEcsrCnlPtr_devi = NULL;
  HANDLE_ERROR(malloc_devi(aEcsrCnlPtr_devi, effM_host[0]+1));
  
  /* System matrix column ids (which are voxel ids). Part of COO representation. */
  int * aVxlId_devi = NULL;
  HANDLE_ERROR(malloc_devi(aVxlId_devi, maxNnz));
  
  /* System matrix values. Part of sparse representations. */
  val_t * aVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(aVal_devi, maxNnz));
  
  /* Number of non-zeros in system matrix. */
  int nnz_host[1];
  int * nnz_devi = NULL;
  HANDLE_ERROR(malloc_devi<int>(nnz_devi, 1));
  
  
  /* X-LIKE VECTORS */
  
  /* Density guess */
  val_t * x_host = new val_t[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) { x_host[i] = 1.; }
  val_t * x_devi = NULL;
  HANDLE_ERROR(malloc_devi(x_devi, VGRIDSIZE));
  HANDLE_ERROR(memcpyH2D(x_devi, x_host, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  do {
    val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
    scales<val_t>(x_devi, (1./norm), VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
  } while(false);
  
  /* Intermediate density guess */
  val_t * xx_host = new val_t[VGRIDSIZE];
  val_t * xx_devi = NULL;
  HANDLE_ERROR(malloc_devi(xx_devi, VGRIDSIZE));
  
  /* Correction */
  val_t * c_devi = NULL;
  HANDLE_ERROR(malloc_devi(c_devi, VGRIDSIZE));
  
  /* Sensitivity */
  val_t * s_devi = NULL;
  HANDLE_ERROR(malloc_devi(s_devi, VGRIDSIZE));
  
  
  /* Y-LIKE VECTORS */
  
  /* Ones */
  val_t * oneVal_host = new val_t[effM_host[0]];
  for(int i=0; i<effM_host[0]; i++) { oneVal_host[i] = 1.; }
  val_t * oneVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(oneVal_devi, effM_host[0]));
  HANDLE_ERROR(memcpyH2D(oneVal_devi, oneVal_host, effM_host[0]));
  
  /* Simulated measurement vector */
  val_t * yTildeVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(yTildeVal_devi, effM_host[0]));
  
  /* "Error" */
  val_t * eVal_devi = NULL; 
  HANDLE_ERROR(malloc_devi(eVal_devi, effM_host[0]));
  
  
  /* OTHER */
  
  /* Handle to cuSPARSE library context. */
  cusparseHandle_t handle = NULL;
  
  /* Matrix descriptor. Used with cuSPARSE library. */
  cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  
  /* Scalar factors for use in matrix vector multiplications */
  val_t alpha = 1.;
  val_t beta  = 0.;
  
  
  /* SENSITIVITY */
  
  /**  @todo getSensitivity(); */
  val_t s_host[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) { s_host[i] = 1.; }
  HANDLE_ERROR(memcpyH2D(s_devi, s_host, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  
  /* RECO ITERATIONS */
  
  /* Get system matrix */
  systemMatrixCalculation<val_t> (
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
        nnz_devi,
        aCnlId_devi, aCsrCnlPtr_devi,
        yRowId_devi, effM_host,
        handle);
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(memcpyD2H(nnz_host, nnz_devi, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());

  /* Matrix vector multiplication to simulate measurement */
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        effM_host[0], VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        x_devi, &beta, yTildeVal_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  std::cout << "sum(yTildeVal_devi):" << sum<val_t>(yTildeVal_devi, effM_host[0]) << std::endl;
  
  /* Division to get "error" */
  divides<val_t>(eVal_devi, yVal_devi, yTildeVal_devi, effM_host[0]);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Matrix vector multiplication to backproject error on grid */
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_TRANSPOSE,
        effM_host[0], VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        eVal_devi, &beta, c_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Apply correction */
  dividesMultiplies<val_t>(xx_devi, x_devi, c_devi, s_devi, VGRIDSIZE);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  /* Normalize */
//  val_t norm = sum<val_t>(xx_devi, VGRIDSIZE);
//  scales<val_t>(density_Guess)
  
  /* Copy back to host */
  HANDLE_ERROR(memcpyD2H(xx_host, xx_devi, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Write to file */
  writeDensity_HDF5(xx_host, on, grid);

  return 0;
}

