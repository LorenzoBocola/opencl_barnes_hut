#include "bh.h"

#include <stdio.h>
#include <math.h>

#define SQR(x) ((x)*(x))
#define G 0.001

oct_tree *newtree(cl_float3 *pos, cl_float *masses, cl_float3 *coms,cl_int *tree,int max_nodes){
    oct_tree *t=malloc(sizeof(oct_tree));
    if(t==NULL){
        fprintf(stderr,"Error: could not allocate memory for oct tree\n");
        exit(1);
    }
    t->tree=tree;
    for(int i=0;i<9;i++){
        t->tree[i]=-1;
    }
    t->nodes_num=1;
    t->leaves=pos;
    t->masses=masses;
    t->coms=coms;
    t->max_nodes=max_nodes;
    return t;
};

void print_node(oct_tree *t, int node){
    printf("NODE %d\n",node);
        printf("LEAF: %d\n",t->tree[9*node]);
        printf("MASS: %f\n",t->masses[node]);
        printf("COM: (%f,%f,%f)\n",t->coms[node].x,t->coms[node].y,t->coms[node].z);
        for(int j=0;j<8;j++){
            printf("\tCHILD %d: %d\n",j,t->tree[9*node+j+1]);
        }
}

int allocate_node(oct_tree *t){
    int index=t->nodes_num*9;
    t->nodes_num++;
    if(t->nodes_num>=t->max_nodes){
        fprintf(stderr,"Error: oct tree overflow\n");
        exit(1);
    }
    for(int i=0;i<9;i++){
        t->tree[index+i]=-1;
    }
    return t->nodes_num-1;
}

void add_leaf_(oct_tree *t,int cell,int depth,cl_float3 pos,int leaf,cl_float3 box_corner){
    if(pos.x>XMAX||pos.x<XMIN||pos.y>YMAX||pos.y<YMIN||pos.z>ZMAX||pos.z<ZMIN){
        return;
    }
    
    cl_float3 box_corner_=box_corner;
    if(depth>MAX_DEPTH){
        fprintf(stderr,"max depth exceeded\n");
        exit(1);
    }
    if(leaf==-1){
        leaf=t->leaves_num++;
    }
    int index=cell*9;//index of the cell in the t->tree array

    float halfx=(XMAX-XMIN)/(2<<depth);//width of the cell's subcells bounding box
    float halfy=(YMAX-YMIN)/(2<<depth);
    float halfz=(ZMAX-ZMIN)/(2<<depth);
    
    /*
        calculating the index of the index of the next cell (tree node) and
        the new bounding box bottom left corner position based on the position
        of the new leaf
    */
    int child_index=index+1;
    if(pos.x-box_corner.x>halfx){
        child_index+=1;
        box_corner_.x+=halfx;
    }if(pos.y-box_corner.y>halfy){
        child_index+=2;
        box_corner_.y+=halfy;
    }if(pos.z-box_corner.z>halfz){
        child_index+=4;
        box_corner_.z+=halfz;
    }

    if(t->tree[child_index]!=-1){
        /*
            if the child cell already exists the function is called again
            with the updated bounding box
        */
        add_leaf_(t,t->tree[child_index],depth+1,pos,leaf,box_corner_);
    }else{
        if(t->tree[index]==-1){
            /*
                if the current cell is not a leaf (contains a single particle)
                a new cell is allocated
            */
            int new_cell=allocate_node(t);
            t->tree[child_index]=new_cell;
            t->tree[new_cell*9]=leaf;
        }else{
            /*
                if the current node is a leaf the function is called again
                twice to move the particle in a deeper subcell and to
                add the new particle
            */
            int old_leaf=t->tree[index];
            t->tree[index]=-1;
            if(t->leaves[old_leaf].x==pos.x&&t->leaves[old_leaf].y==pos.y&&t->leaves[old_leaf].z==pos.z){
                /*
                    The algorithm is not designed to handle multiple particles
                    with the same position
                */
                printf("TWO PARTICLES SHARING THE SAME POSITION (WILL PROBABLY CRASH)\n");
                printf("%d,%f %f %f\n",leaf,pos.x,pos.y,pos.y);
                printf("%d,%f %f %f\n",old_leaf,t->leaves[old_leaf].x,t->leaves[old_leaf].y,t->leaves[old_leaf].y);
            }
            add_leaf_(t,cell,depth,t->leaves[old_leaf],old_leaf,box_corner);
            add_leaf_(t,cell,depth,pos,leaf,box_corner);
        } 
    }
}

void compute_masses(oct_tree *t,float *masses,cl_float3 *pos){
    /*
        Since the parent of a cell always precedes it it is possibile to
        compute the masses and centers of mass of each cell by scanning the
        tree array backwards.
    */
    for(int i=t->nodes_num-1;i>=0;i--){
        //printf("BEFORE PROCESSING\n");
        //print_node(t,i);
        for(int child=0;child<8-1;child++){
            if(t->tree[9*i+1+child]==-1){
                for(int child1=child+1;child1<8;child1++){
                    if(t->tree[9*i+1+child1]!=-1){
                        t->tree[9*i+1+child]=t->tree[9*i+1+child1];
                        t->tree[9*i+1+child1]=-1;
                        break;
                    }
                }
            }
        }
        if(t->tree[9*i]!=-1){
            t->masses[i]=masses[t->tree[9*i]];
            t->coms[i]=pos[t->tree[9*i]];
        }else{
            float mass_sum=0;
            cl_float3 com_sum={0};

            for(int j=0;j<8;j++){
                int subcell=t->tree[9*i+1+j];
                if(subcell!=-1){
                    float mass=t->masses[t->tree[9*i+1+j]];
                    com_sum.x+=mass*t->coms[subcell].x;
                    com_sum.y+=mass*t->coms[subcell].y;
                    com_sum.z+=mass*t->coms[subcell].z;
                    mass_sum+=mass;
                }
            }
            t->coms[i].x=com_sum.x/mass_sum;
            t->coms[i].y=com_sum.y/mass_sum;
            t->coms[i].z=com_sum.z/mass_sum;
            t->masses[i]=mass_sum;
        }
        //printf("AFTER PROCESSING\n");
        //print_node(t,i);
    }
}

#if 0

//the following functions have now been implemented in OpenCL kernels

void compute_accelerationsn2(int id,float *masses,cl_float3 *pos, cl_float3 *acc, int n){
    cl_float3 acc_tmp;
    for(int j=0;j<n;j++){
        if(j==id){
            continue;
        }
        cl_float3 posdiff;
        posdiff.x=pos[j].x-pos[id].x;
        posdiff.y=pos[j].y-pos[id].y;
        posdiff.z=pos[j].z-pos[id].z;
        float distsq=SQR(posdiff.x)+SQR(posdiff.y)+SQR(posdiff.z);
        acc_tmp.x+=posdiff.x*masses[j]*G/distsq/sqrt(distsq);
        acc_tmp.y+=posdiff.y*masses[j]*G/distsq/sqrt(distsq);
        acc_tmp.z+=posdiff.z*masses[j]*G/distsq/sqrt(distsq);
        //printf("1>>DIST: %f\n",sqrt(distsq));
    }
    acc[id+0]=acc_tmp;
}

void compute_accelerationsln_norec(int id, oct_tree *t,float *masses,cl_float3 *pos,cl_float3 *accv){
    int cell=0;
    //printf("CELL %d\n",cell);
    //printf("2>>DIST: %f\n",sqrt(distsq));
    int cell_stack[20]={0};
    int subcell_stack[20]={0};
    int stackind=0;
    int counter=0;
    while(stackind>=0){
        counter++;
        float distsq=SQR(t->coms[cell].x-pos[id].x)+SQR(t->coms[cell].y-pos[id].y)+SQR(t->coms[cell].z-pos[id].z);
        float precisionsq=SQR((XMAX-XMIN)/(1<<stackind))/distsq;
        if(precisionsq<SQR(PRECISION)||t->tree[9*cell]!=-1){
            //compute acceleration
            if(t->tree[9*cell]!=id){
                cl_float3 posdiff;
                posdiff.x=t->coms[cell].x-pos[id].x;
                posdiff.y=t->coms[cell].y-pos[id].y;
                posdiff.z=t->coms[cell].z-pos[id].z;
                accv->x+=posdiff.x*t->masses[cell]*G/distsq/sqrt(distsq);
                accv->z+=posdiff.y*t->masses[cell]*G/distsq/sqrt(distsq);
                accv->y+=posdiff.z*t->masses[cell]*G/distsq/sqrt(distsq);
            }
            cell=cell_stack[stackind];
            stackind--;
            subcell_stack[stackind]++;
        }else{
            int subcell=t->tree[9*cell+1+subcell_stack[stackind]];
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
}
#endif