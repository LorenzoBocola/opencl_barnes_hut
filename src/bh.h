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

typedef struct _oct_tree2_struct{
    int nodes_num;
    int leaves_num;
    cl_float3 *leaves;
    cl_int *tree_children;
    cl_int *tree_leaves;
    cl_float *masses;
    cl_float3 *coms;
    int max_nodes;
    ot_bbox bbox;
}oct_tree;

oct_tree *newtree(cl_float3 *pos, cl_float *masses, cl_float3 *coms,cl_int *tree_children,cl_int *tree_leaves,int max_nodes,ot_bbox bbox);
//void add_leaf_(oct_tree *t,int cell,int depth,cl_float3 pos,int leaf,cl_float3 box_corner);
void fill_tree(oct_tree *t,cl_float3 *leaves_pos,int leaves_num);
void summarize_tree(oct_tree *t,float *masses,cl_float3 *pos);

//#define add_leaf(tree,id,pos) add_leaf_(tree,0,0,pos,id,(cl_float3){tree->bbox.xrange.s0,tree->bbox.yrange.s0,tree->bbox.zrange.s0})

#endif