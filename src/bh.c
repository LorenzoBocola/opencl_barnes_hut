#include "bh.h"

#include <stdio.h>

#define SQR(x) ((x)*(x))
#define G 0.001

oct_tree *newtree(
    cl_float3 *pos,
    cl_float *masses,
    cl_float3 *coms,
    cl_int *tree_children,
    cl_int *tree_leaves,
    int max_nodes,
    ot_bbox bbox
){
    oct_tree *t=malloc(sizeof(oct_tree));
    if(t==NULL){
        fprintf(stderr,"Error: could not allocate memory for oct tree\n");
        exit(1);
    }

    t->tree_children=tree_children;
    t->tree_leaves=tree_leaves;
    for(int i=0;i<8;i++){
        t->tree_children[i]=-1;
    }
    t->tree_leaves[0]=-1;

    t->nodes_num=1;
    t->leaves=pos;
    t->masses=masses;
    t->coms=coms;
    t->max_nodes=max_nodes;
    t->bbox=bbox;
    return t;
};

void print_node(oct_tree *t, int node){
    printf("NODE %d\n",node);
        printf("LEAF: %d\n",t->tree_leaves[node]);
        printf("MASS: %f\n",t->masses[node]);
        printf("COM: (%f,%f,%f)\n",t->coms[node].x,t->coms[node].y,t->coms[node].z);
        for(int j=0;j<8;j++){
            printf("\tCHILD %d: %d\n",j,t->tree_children[8*node+j]);
        }
}

int alocate_node(oct_tree *t){
    int index=t->nodes_num*8;
    t->nodes_num++;
    if(t->nodes_num>=t->max_nodes){
        fprintf(stderr,"Error: oct tree overflow\n");
        exit(1);
    }
    for(int i=0;i<8;i++){
        t->tree_children[index+i]=-1;
    }
    t->tree_leaves[t->nodes_num]=-1;
    return t->nodes_num-1;
}

void add_leaf_(oct_tree *t,int cell,int depth,cl_float3 pos,int leaf,cl_float3 box_corner){
    if(pos.x>t->bbox.xrange.s1||pos.x<t->bbox.xrange.s0||pos.y>t->bbox.yrange.s1||pos.y<t->bbox.yrange.s0||pos.z>t->bbox.zrange.s1||pos.z<t->bbox.zrange.s0){
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
    //int index=cell*9;//index of the cell in the t->tree array

    float halfx=(t->bbox.xrange.s1-t->bbox.xrange.s0)/(2<<depth);//width of the cell's subcells bounding box
    float halfy=(t->bbox.yrange.s1-t->bbox.yrange.s0)/(2<<depth);
    float halfz=(t->bbox.zrange.s1-t->bbox.zrange.s0)/(2<<depth);
    
    /*
        calculating the index of the index of the next cell (tree node) and
        the new bounding box bottom left corner position based on the position
        of the new leaf
    */
    int child_index=8*cell;
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

    if(t->tree_children[child_index]!=-1){
        /*
            if the child cell already exists the function is called again
            with the updated bounding box
        */
        add_leaf_(t,t->tree_children[child_index],depth+1,pos,leaf,box_corner_);
    }else{
        if(t->tree_leaves[cell]==-1){
            /*
                if the current cell is not a leaf (contains a single particle)
                a new cell is allocated
            */
            int new_cell=alocate_node(t);
            t->tree_children[child_index]=new_cell;
            t->tree_leaves[new_cell]=leaf;
        }else{
            /*
                if the current node is a leaf the function is called again
                twice to move the particle in a deeper subcell and to
                add the new particle
            */
            int old_leaf=t->tree_leaves[cell];
            t->tree_leaves[cell]=-1;
            if(t->leaves[old_leaf].x==pos.x&&t->leaves[old_leaf].y==pos.y&&t->leaves[old_leaf].z==pos.z){
                /*
                    The algorithm is not designed to handle multiple particles
                    with the same position
                */
                printf("TWO PARTICLES SHARING THE SAME POSITION (WILL PROBABLY CRASH)\n");
                printf("%d,%f %f %f\n",leaf,pos.x,pos.y,pos.z);
                printf("%d,%f %f %f\n",old_leaf,t->leaves[old_leaf].x,t->leaves[old_leaf].y,t->leaves[old_leaf].z);
            }
            add_leaf_(t,cell,depth,t->leaves[old_leaf],old_leaf,box_corner);
            add_leaf_(t,cell,depth,pos,leaf,box_corner);
        } 
    }
}

void fill_tree(oct_tree *t,cl_float3 *leaves_pos,int leaves_num){
    for(int i=0;i<leaves_num;i++){
        add_leaf_(t,0,0,leaves_pos[i],i,(cl_float3){t->bbox.xrange.s0,t->bbox.yrange.s0,t->bbox.zrange.s0});
    }
}

void summarize_tree(oct_tree *t,float *masses,cl_float3 *pos){
    /*
        Since the parent of a cell always precedes it it is possibile to
        compute the masses and centers of mass of each cell by scanning the
        tree array backwards.
    */
    for(int i=t->nodes_num-1;i>=0;i--){
        //printf("BEFORE PROCESSING\n");
        //print_node(t,i);
        for(int child=0;child<8-1;child++){
            if(t->tree_children[8*i+child]==-1){
                for(int child1=child+1;child1<8;child1++){
                    if(t->tree_children[8*i+child1]!=-1){
                        t->tree_children[8*i+child]=t->tree_children[8*i+child1];
                        t->tree_children[8*i+child1]=-1;
                        break;
                    }
                }
            }
        }
        if(t->tree_leaves[i]!=-1){
            t->masses[i]=masses[t->tree_leaves[i]];
            t->coms[i]=pos[t->tree_leaves[i]];
        }else{
            float mass_sum=0;
            cl_float3 com_sum={0};

            for(int j=0;j<8;j++){
                int subcell=t->tree_children[8*i+j];
                if(subcell!=-1){
                    float mass=t->masses[t->tree_children[8*i+j]];
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