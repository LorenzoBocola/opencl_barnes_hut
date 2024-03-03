#ifndef _BH_H_
#define _BH_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define MAX_DEPTH 50
#define PRECISION 0.1

typedef struct _ot_bbox_struct{
    cl_float2 xrange;
    cl_float2 yrange;
    cl_float2 zrange;
}ot_bbox;

typedef struct _bh_tree2_struct{
    int nodes_num;
    int leaves_num;
    cl_float3 *leaves;
    cl_int *tree_children;
    cl_int *tree_leaves;
    cl_float *masses;
    cl_float3 *coms;
    int max_nodes;
    ot_bbox bbox;
}bh_tree;

bh_tree *newtree(cl_float3 *pos, cl_float *masses, cl_float3 *coms,cl_int *tree_children,cl_int *tree_leaves,int max_nodes,ot_bbox bbox);
void fill_tree(bh_tree *t,cl_float3 *leaves_pos,int leaves_num);
void summarize_tree(bh_tree *t,float *masses,cl_float3 *pos);

#endif//_BH_H_