#ifndef _BH_H_
#define _BH_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define N 10000000

#define XMIN -10.
#define XMAX 10.
#define YMIN -10.
#define YMAX 10.
#define ZMIN -10.
#define ZMAX 10.

#define MAX_DEPTH 50
#define PRECISION 0.1

typedef struct _oct_tree_struct{
    int nodes_num;
    int leaves_num;
    cl_float3 *leaves;
    cl_int *tree;
    cl_float *masses;
    cl_float3 *coms;
    int max_nodes;
}oct_tree;

oct_tree *newtree(cl_float3 *pos, cl_float *masses, cl_float3 *coms,cl_int *tree,int max_nodes);
void add_leaf_(oct_tree *t,int cell,int depth,cl_float3 pos,int leaf,cl_float3 box_corner);
void compute_masses(oct_tree *t,float *masses,cl_float3 *pos);
#if 0
void compute_accelerationsn2(int id,float *masses,cl_float3 *pos, cl_float3 *acc, int n);
void compute_accelerationsln_norec(int id, oct_tree *t,float *masses,cl_float3 *pos,cl_float3 *accv);
#endif

#define add_leaf(tree,id,pos) add_leaf_(tree,0,0,pos,id,(cl_float3){XMIN,YMIN,ZMIN})

#endif