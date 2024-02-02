#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable

#define SQR(x) ((x)*(x))

#define G 0.001f

#define XMIN -10.
#define XMAX 10.
#define YMIN -10.
#define YMAX 10.
#define ZMIN -10.
#define ZMAX 10.

#define PRECISION 0.75

#define MAX_DEPTH 50

__kernel void compute_accelerations(
    __global const float *masses,
    __global const float3 *pos,
    __global float3 *acc,
    const int len
){
    int id=get_global_id(0);
    if(id>=len){
        return;
    }
    
    //float selfmass=masses[id];
    float3 selfpos=pos[id];
    float3 acctmp=0;

    //computing acceleration for each other particle
    for(int i=0;i<len;i++){
        if(i==id){
            continue;
        }
        float3 unitdist=pos[i]-selfpos;
        float distsq=dot(unitdist,unitdist);
        unitdist=normalize(unitdist);
        acctmp+=unitdist*G*masses[i]/distsq;
    }

    acc[id]=acctmp;
}

__kernel void verlet_new_pos(
    __global float3 *pos,
    __global const float3 *vel,
    __global const float3 *acc,
    const int len,
    const float dt
){
    int id=get_global_id(0);
    if(id>=len){
        return;
    }
    pos[id]=pos[id]+vel[id]*dt+acc[id]*SQR(dt)/2;
}

__kernel void verlet_new_vel(
    __global float3 *vel,
    __global const float3 *acc_old,
    __global const float3 *acc,
    const int len,
    const float dt
){
    int id=get_global_id(0);
    if(id>=len){
        return;
    }
    vel[id]=vel[id]+dt*(acc_old[id]+acc[id])/2;
}

__kernel void barnes_hut_accelerations(
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
    int stackind=0;
    int precise_enough=0;
    float3 posdiff;
    float distsq;
    int subcell;
    while(stackind>=0){
        //float3 posdiff=tree_coms[cell]-pos;
        //float distsq=dot(posdiff,posdiff);
        //float precisionsq=SQR((XMAX-XMIN)/(1<<stackind))/distsq;
        int particleid=tree[9*cell];
        if(precise_enough||particleid!=-1){
            //compute acceleration
            if(particleid!=id){
                acctmp+=rsqrt(distsq)*posdiff*tree_masses[cell]*G/distsq;
            }
            precise_enough=0;
            subcell=-1;
        }else{
            subcell=tree[9*cell+1+subcell_stack[stackind]];
        }
        if(subcell!=-1&&subcell_stack[stackind]<8){
            //going one cell deeper
            stackind++;
            cell_stack[stackind]=cell;
            cell=subcell;
            subcell_stack[stackind]=0;
            posdiff=tree_coms[cell]-pos;
            distsq=dot(posdiff,posdiff);
            precise_enough=SQR((XMAX-XMIN)/(1<<stackind))/distsq<SQR(PRECISION);
        }else{
            cell=cell_stack[stackind];
            stackind--;
            subcell_stack[stackind]++;
        }
    }

    acc[id]=acctmp;
}

/*//old version
__kernel void barnes_hut_accelerations(
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
    int stackind=0;
    int precise_enough=0;
    float3 posdiff;
    float distsq;
    while(stackind>=0){
        //float3 posdiff=tree_coms[cell]-pos;
        //float distsq=dot(posdiff,posdiff);
        //float precisionsq=SQR((XMAX-XMIN)/(1<<stackind))/distsq;
        int particleid=tree[9*cell];
        if(precise_enough||particleid!=-1){
            //compute acceleration
            if(particleid!=id){
                acctmp+=normalize(posdiff)*tree_masses[cell]*G/distsq;
            }
            subcell_stack[stackind]=8;
            precise_enough=0;
        }else{
            int subcell=tree[9*cell+1+subcell_stack[stackind]];
            if(subcell!=-1){
                //going one cell deeper
                stackind++;
                cell_stack[stackind]=cell;
                cell=subcell;
                subcell_stack[stackind]=0;
                posdiff=tree_coms[cell]-pos;
                distsq=dot(posdiff,posdiff);
                precise_enough=SQR((XMAX-XMIN)/(1<<stackind))/distsq<SQR(PRECISION);
            }else{
                subcell_stack[stackind]++;
            }
        }
        while(subcell_stack[stackind]>=8){
            cell=cell_stack[stackind];
            stackind--;
            subcell_stack[stackind]++;
        }
    }

    acc[id]=acctmp;
}

//older version
__kernel void barnes_hut_accelerations(
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
    int stackind=0;
    while(stackind>=0){
        float3 posdiff=tree_coms[cell]-pos;
        float distsq=dot(posdiff,posdiff);
        float precisionsq=SQR((XMAX-XMIN)/(1<<stackind))/distsq;
        int particleid=tree[9*cell];
        if(precisionsq<SQR(PRECISION)||particleid!=-1){
            //compute acceleration
            if(particleid!=id){
                acctmp+=normalize(posdiff)*tree_masses[cell]*G/distsq;
            }
            subcell_stack[stackind]=8;
        }else{
            int subcell=tree[9*cell+1+subcell_stack[stackind]];
            if(subcell!=-1){
                stackind++;
                cell_stack[stackind]=cell;
                cell=subcell;
                subcell_stack[stackind]=0;
            }else{
                subcell_stack[stackind]++;
            }
        }
        while(subcell_stack[stackind]>=8){
            cell=cell_stack[stackind];
            stackind--;
            subcell_stack[stackind]++;
        }
    }

    acc[id]=acctmp;
}
*/

__kernel void count_particles(
    __global int *hist,
    __global const float3 *pos,
    const int2 size,
    const float2 box_min,
    const float2 box_max,
    const int len
){
    int id=get_global_id(0);
    if(id>len){
        return;
    }
    float3 selfpos=pos[id];

    int x=(int)(size.x*(selfpos.x-box_min.x)/(box_max.x-box_min.x));
    int y=(int)(size.y*(selfpos.y-box_min.y)/(box_max.y-box_min.y));

    if(x<0||x>=size.x||y<0||y>=size.y){
        return;
    }

    atomic_inc(hist+size.x*y+x);
}

__kernel void process_hist(
    __global const int *hist,
    __global float *out,
    const int2 size,
    const float2 bbox_min,
    const float2 bbox_max,
    const int particles
){
    int2 pos={get_global_id(0),get_global_id(1)};
    //if(pos.x==0&&pos.y==0)printf("TEST\n");
    if(pos.x>=size.x||pos.y>=size.y){
        return;
    }
    int mask_radius=2;
    float mask[5][5]={
        {0,0,0.25,0,0},
        {0,0.25,0.5,0.25,0},
        {0.25,0.5,1,0.5,0.25},
        {0,0.25,0.5,0.25,0},
        {0,0,0.25,0,0}
    };
    /*
    float mask[5][5]={
        {0,0,0,0,0},
        {0,0,0.125,0,0},
        {0,0.125,0.5,0.125,0},
        {0,0,0.125,0,0},
        {0,0,0,0,0}
    };
    */
    //printf("%d\n",particles);
    float tmp=0;
    //tmp=hist[size.x*pos.x+pos.y];
    for(int xx=pos.x-mask_radius;xx<pos.x+mask_radius+1;xx++){
        for(int yy=pos.y-mask_radius;yy<pos.y+mask_radius+1;yy++){
            if(xx<0||yy<0||xx>=size.x||yy>=size.y){
                continue;
            }
            tmp+=hist[size.x*yy+xx]*mask[xx-pos.x+mask_radius][yy-pos.y+mask_radius];
        }
    }
    float unprocessed=tmp*((size.x*size.y)/((bbox_max.x-bbox_min.x)*(bbox_max.y-bbox_min.y)))/particles;
    out[size.x*(size.y-1-pos.y)+pos.x]=0.5*log(1+0.5*unprocessed);
    //out[size.x*(size.y-1-pos.y)+pos.x]=pos.x%2;
    //out[size.x*pos.y+pos.x]=1;
    //out[size.x*pos.x+pos.y]=hist[size.x*pos.x+pos.y];
}

__kernel void f2rgb(
    __global float *in,
    __global char *out
){
    int2 pixel={get_global_id(0),get_global_id(1)};
    int id=1920*pixel.y+pixel.x;
    out[3*id+0]=255*clamp(in[id],0.f,1.f);
    out[3*id+1]=255*clamp(in[id],0.f,1.f);
    out[3*id+2]=255*clamp(in[id],0.f,1.f);
    //out[3*id+0]=255*in[id];
    //out[3*id+1]=255*in[id];
    //out[3*id+2]=255*in[id];
}