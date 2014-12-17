/* 
 * File:   example_condense_defines.h
 * Author: malte
 *
 * Created on 26. November 2014, 15:53
 */

#ifndef EXAMPLE_CONDENSE_DEFINES_H
#define	EXAMPLE_CONDENSE_DEFINES_H



//#define NBLOCKS     1
//#define TPB         1
//#define SEED        1234
//#define THRESHOLD   0.5
//#define NTEST       10

#define FIRST_ID    0
#define LATER_ID    (NTEST % (NBLOCKS*TPB))
#define FIRST_LEN   ((NTEST / (NBLOCKS*TPB)) + 1)
#define LATER_LEN   (NTEST / (NBLOCKS*TPB))
#define SIZE        NTEST
  
//#define BOATSIZE 4
//#define TRUCKSIZE   2
//#define RINGSIZE    (2 * TRUCKSIZE)
//#define RINGSIZE    ((TPB*BOATSIZE) < (2*TRUCKSIZE) ? (2*TRUCKSIZE) : (TPB*BOATSIZE))
#define RINGSIZE    (TRUCKSIZE - 1 + (TPB*BOATSIZE))


  
#endif	/* EXAMPLE_CONDENSE_DEFINES_H */

