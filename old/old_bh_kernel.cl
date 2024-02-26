#define SQR(x) ((x)*(x))

#define G 0.001f

#define XMIN -10.f
#define XMAX 10.f
#define YMIN -10.f
#define YMAX 10.f
#define ZMIN -10.f
#define ZMAX 10.f

#define PRECISION 0.75f

#define MAX_DEPTH 50

__kernel void barnes_hut_accelerations_old(
    __global float3 *acc,
    __global const float3 *positions,
    __global const float3 *tree_coms,
    __global const float *tree_masses,
    __global const int *tree,
    const int len
){
    int id=get_global_id(0);

    if(id>=len){
        return;
    }

    float3 acctmp=0;

    float3 pos=positions[id];

    int cell=0;
    int cell_stack[MAX_DEPTH]={0};
    int subcell_stack[MAX_DEPTH]={0};
    int depth=0;
    int precise_enough=0;
    float3 posdiff;
    float distsq;
    int subcell;
    while(depth>=0){
        //float3 posdiff=tree_coms[cell]-pos;
        //float distsq=dot(posdiff,posdiff);
        //float precisionsq=SQR((XMAX-XMIN)/(1<<depth))/distsq;
        int particleid=tree[9*cell];
        if(precise_enough||particleid!=-1){
            //computing acceleration
            if(particleid!=id){
                acctmp+=rsqrt(distsq)*posdiff*tree_masses[cell]*G/distsq;
            }
            precise_enough=0;
            subcell=-1;
        }else{
            subcell=tree[9*cell+1+subcell_stack[depth]];
        }
        if(subcell!=-1&&subcell_stack[depth]<8){
            //increasing depth
            depth++;
            cell_stack[depth]=cell;
            cell=subcell;
            subcell_stack[depth]=0;
            posdiff=tree_coms[cell]-pos;
            distsq=dot(posdiff,posdiff);
            precise_enough=SQR((XMAX-XMIN)/(1<<depth))/distsq<SQR(PRECISION);
        }else{
            //decreasing depth
            cell=cell_stack[depth];
            depth--;
            subcell_stack[depth]++;
        }
        //mem_fence(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }

    acc[id]=acctmp;
}