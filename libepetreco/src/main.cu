#include "CUDA_HandleError.hpp"
#include "ChordsCalc_kernel2.cu"
#include "visualization.hpp"

#include <iostream>
#include <sstream>

int main()
{
  // Host memory allocation, initialization
  int   gridN_host[] = {GRIDNX, GRIDNY, GRIDNZ};
  val_t gridO_host[] = {-1.5, -0.5, -2.0};
  val_t gridD_host[] = {1.0, 1.0, 1.0};

  int linearChannelId = 1;

  val_t chords_host  [VGRIDSIZE*NTHREADS*NBLOCKS];
  for(int i=0; i<VGRIDSIZE*NBLOCKS; i++)
    chords_host[i]=0;

  val_t rays_host[6*NBLOCKS*NTHREADRAYS];
  
  // Visualize grid
  PlyGrid<Vertex> grid("",
                       Vertex(gridO_host[0], gridO_host[1], gridO_host[2]),
                       gridN_host[0]+1, gridN_host[1]+1, gridN_host[2]+1,
                       gridD_host[0],   gridD_host[1],   gridD_host[2]);
  PlyWriter writer("ChordsCalc_kernel2_grid.ply");
  writer.write(grid);
  writer.close();
  
  // Visualize det0
  int   det0N[] = {1, N0Y, N0Z};
  val_t det0C[] = {POS0X, 0, 0};
  val_t detD[]  = {SEGX, SEGY, SEGZ};
  BetaPlyGrid det0("", det0C, detD, det0N, BetaPlyGrid::AT_CENTER);
  PlyWriter det0Writer("ChordsCalc_kernel2_det0.ply");
  det0Writer.write(det0);
  det0Writer.close();

  // Visualize det1
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  BetaPlyGrid det1("", det1C, detD, det1N, BetaPlyGrid::AT_CENTER);
  PlyWriter det1Writer("ChordsCalc_kernel2_det1.ply");
  det1Writer.write(det1);
  det1Writer.close();
  
  // Device memory allocation
  val_t * chords_devi;
  HANDLE_ERROR( cudaMalloc((void**)&chords_devi,
                VGRIDSIZE*NBLOCKS*sizeof(val_t)) );
  
  val_t * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi,
                6*NBLOCKS*NTHREADRAYS*sizeof(val_t)) );
  
  int *   gridN_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridN_devi,
                3*sizeof(int)) );
  
  val_t * gridO_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridO_devi,
                3*sizeof(val_t)) );
  
  val_t * gridD_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridD_devi,
                3*sizeof(val_t)) );
  
  // Copy host to device
  HANDLE_ERROR( cudaMemcpy(chords_devi, chords_host,
                           VGRIDSIZE*NBLOCKS*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridN_devi, gridN_host,
                           3*sizeof(int),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridO_devi, gridO_host,
                           3*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridD_devi, gridD_host,
                           3*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  
  // Run kernel
  chordsCalc<<<NBLOCKS,1>>>(chords_devi, rays_devi,
                                   gridO_devi, gridD_devi, gridN_devi );
  HANDLE_ERROR( cudaDeviceSynchronize() );
  
  // Copy results device to host
  HANDLE_ERROR( cudaMemcpy(chords_host, chords_devi,
                           VGRIDSIZE*NBLOCKS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(rays_host, rays_devi,
                           6*NBLOCKS*NTHREADRAYS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  // Print SM
  for(int i=0; i<NBLOCKS; i++)
  {
    std::cout << "### channel " << i << ":" << std::endl;
    for(int j=0; j<VGRIDSIZE; j++)
    {
      std::cout << "voxel " << j << ": "
                << chords_host[i*VGRIDSIZE+j]/(NTHREADRAYS)
                << std::endl;
    }
    std::cout << std::endl;
  }

  // Visualize rays
  for(int i=0; i<NBLOCKS; i++)
  {
    BetaCompositePlyGeom compositeLines("");
    BetaPlyLine lines[NTHREADRAYS];
    for(int idRay=0; idRay<NTHREADRAYS; idRay++)
    {
      lines[idRay] = BetaPlyLine("", &rays_host[6*(i*NTHREADRAYS + idRay)]);
      compositeLines.add(&lines[idRay]);
    }

    std::stringstream fn("");
    fn << "ChordsCalc_kernel2_rays-"
       << i << ".ply";
    PlyWriter raysWriter(fn.str());
    raysWriter.write(compositeLines);
    raysWriter.close();
  }
  
  // Clean up
  cudaFree(chords_devi);
  cudaFree(gridN_devi);
  cudaFree(gridO_devi);
  cudaFree(gridD_devi);

  return 0;
}
