#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "clutils.h"
#include "log.h"

#include "bh.h"

#ifndef M_PI
#define M_PI 0
//#error "M_PI undefined"
#endif

#define PARTICLES 1000000
#define STEPS 1000

#define PRINT_VECTOR(fptr,s,vec) fprintf(fptr,"%f,%f,%f"s,(vec).x,(vec).y,(vec).z);
#define SCAN_VECTOR(fptr,s,vecp) fscanf(fptr,"%f,%f,%f"s,&((vecp)->x),&((vecp)->y),&((vecp)->z));

#define GRID_XSIZE 100
#define GRID_YSIZE 100
#define GRID_ZSIZE 1
#define XMIN -10.
#define XMAX 10.
#define YMIN -10.
#define YMAX 10.

#define TREE_SIZE (3*PARTICLES)

#define BH

#define HIST_WIDTH 1920
#define HIST_HEIGHT 1080


FILE *secure_fopen(const char *filename, const char *mode){
    FILE *fptr=fopen(filename,mode);
    if(fptr==NULL){
        fprintf(stderr,"Error: file \"%s\" could not be opened\n",filename);
        exit(1);
    }
    return fptr;
}

void dump_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len){
    for(int i=0;i<len;i++){
        fprintf(fptr,"%f,",color[i]);
        fprintf(fptr,"%f,",masses[i]);
        PRINT_VECTOR(fptr,",",pos[i]);
        PRINT_VECTOR(fptr,",",vel[i]);
        PRINT_VECTOR(fptr,",",acc_old[i]);
        PRINT_VECTOR(fptr,"\n",acc[i]);
    }
}

void load_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len){
    for(int i=0;i<len;i++){
        int ret;
        ret=fscanf(fptr,"%f,",color+i);
        ret=fscanf(fptr,"%f,",masses+i);
        ret=SCAN_VECTOR(fptr,",",pos+i);
        ret=SCAN_VECTOR(fptr,",",vel+i);
        ret=SCAN_VECTOR(fptr,",",acc_old+i);
        ret=SCAN_VECTOR(fptr,"\n",acc+i);
        if(ret==EOF){
            break;
        }
    }
}

void init_state(cl_float *masses,float *color,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len){
#if 1
    for(int i=0;i<len;i++){
        masses[i]=2000./PARTICLES;
        acc[i]=(cl_float3){0,0,0};
        acc_old[i]=(cl_float3){0,0,0};
        if(i<len/4){
            color[i]=0;
            float r=0.5*drand48();
            float theta=drand48()*2*M_PI;
            pos[i]=(cl_float3){r*cos(theta)+3,r*sin(theta)+0.75,0.05*(2*drand48()-1)};
            vel[i]=(cl_float3){-sqrt(r)*sin(theta)-1,sqrt(r)*cos(theta)};
        }else{
            color[i]=1;
            float r=drand48();
            float theta=drand48()*2*M_PI;
            pos[i]=(cl_float3){r*cos(theta)-1,r*sin(theta),0.1*(2*drand48()-1)};
            vel[i]=(cl_float3){-1.5*sqrt(r)*sin(theta)+0.333,1.5*sqrt(r)*cos(theta),0};
        }
    }
#else
    for(int i=0;i<len;i++){
        color[i]=1;
        masses[i]=2000./PARTICLES;
        acc[i]=(cl_float3){0,0,0};
        acc_old[i]=(cl_float3){0,0,0};
        pos[i]=(cl_float3){drand48()*4-2,drand48()*4-2,0.1*drand48()};
        vel[i]=(cl_float3){0,0,0};
    }
#endif
}

void dump_positions(FILE *fptr,cl_float3 *pos,float *color,cl_int len){
    for(int i=0;i<len;i++){
        PRINT_VECTOR(fptr,",",pos[i]);
        fprintf(fptr,"%f\n",color[i]);
    }
}

int main(int argc, char *argv[]){
    srand48(time(NULL));

    cl_int zero=0;

    FILE *logptr=secure_fopen("nbody.log","w");
    logtimer *logt=create_timer(logptr);
    //log_partial(logt,"start");

    cl_int len;
    int first_frame=0;
    if(argc<2){
        len=PARTICLES;
    }else{
        if(argc<4){
            fprintf(stderr,"Error: invalid number of arguments\n");
            exit(1);
        }
        len=atoi(argv[2]);
        first_frame=atoi(argv[3]);
    }

    cl_float dt=0.005;

    cl_int hist_width=HIST_WIDTH;
    cl_int hist_height=HIST_HEIGHT;

    cl_float2 imagebox_min={-4,-4*0.5625};
    cl_float2 imagebox_max={4,4*0.5625};
    //cl_float2 imagebox_min={XMIN,YMIN};
    //cl_float2 imagebox_max={XMAX,YMAX};

    cl_int2 hist_dimensions={hist_width,hist_height};

    FILE *fptr;
    /*=secure_fopen("nbody.cfg","w");
    fprintf(fptr,"frames=%d",STEPS);
    fclose(fptr);*/

    size_t float_data_size=len*sizeof(cl_float);
    size_t vector_data_size=len*sizeof(cl_float3);

    float *color=malloc(len*sizeof(float));
    cl_float *masses=malloc(len*sizeof(cl_float));
    cl_float3 *pos=malloc(len*sizeof(cl_float3));
    cl_float3 *vel=malloc(len*sizeof(cl_float3));
    cl_float3 *acc_old=malloc(len*sizeof(cl_float3));
    cl_float3 *acc=malloc(len*sizeof(cl_float3));

    size_t tree_coms_size=TREE_SIZE*sizeof(cl_float3);
    size_t tree_masses_size=TREE_SIZE*sizeof(cl_float);
    size_t tree_tree_size=9*TREE_SIZE*sizeof(cl_int);

    size_t hist_size=hist_width*hist_height*sizeof(cl_int);
    size_t hist_f_size=hist_width*hist_height*sizeof(cl_float);
    size_t image_size=hist_width*hist_height*3*sizeof(cl_char);

    cl_float3 *acc_test=malloc(len*sizeof(cl_float3));

    cl_float3 *tree_coms=malloc(TREE_SIZE*sizeof(cl_float3));
    cl_float *tree_masses=malloc(TREE_SIZE*sizeof(cl_float));
    cl_int *tree_tree=malloc(9*TREE_SIZE*sizeof(cl_int));

    cl_int *hist=malloc(hist_size);
    cl_float *hist_f=malloc(hist_f_size);
    cl_char *image=malloc(image_size);

    log_timer(logt,"Host memory allocated",0,0);
    

    if(argc<2){
        //data initialization
        init_state(masses,color,pos,vel,acc_old,acc,len);
    }else{
        fptr=secure_fopen(argv[1],"r");
        load_state(fptr,color,masses,pos,vel,acc_old,acc,len);
        fclose(fptr);
    }

    cl_mem mass_buffer=NULL;
    cl_mem pos_buffer=NULL;
    cl_mem vel_buffer=NULL;
    cl_mem acc_buffer0=NULL;
    cl_mem acc_buffer1=NULL;

    cl_mem hist_buffer=NULL;
    cl_mem hist_f_buffer=NULL;
    cl_mem image_buffer=NULL;

    cl_mem acc_test_buffer=NULL;

    cl_mem tree_coms_buffer=NULL;
    cl_mem tree_masses_buffer=NULL;
    cl_mem tree_tree_buffer=NULL;


    cl_mem *acc_bufferp[2]={&acc_buffer0,&acc_buffer1};

	cl_device_id device_id = NULL;

	cl_context context = NULL;
	cl_kernel kernel_acc = NULL;
    cl_kernel kernel_vpos = NULL;
    cl_kernel kernel_vvel=NULL;

    cl_kernel kernel_bh=NULL;

    cl_kernel kernel_hist=NULL;
    cl_kernel kernel_hist_processing=NULL;
    cl_kernel kernel_f2rgb=NULL;

	cl_program program = NULL;


	cl_command_queue command_queue = NULL;
	cl_int ret;


    create_context(&context,&device_id);
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    testerror(ret,"Failed to create command queue ##");
	// Memory Buffer
	mass_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, float_data_size, NULL, &ret);
    pos_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    vel_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    acc_buffer0 = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    acc_buffer1 = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);

    acc_test_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);

    hist_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, hist_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    hist_f_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, hist_f_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    image_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, image_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    
    tree_coms_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_coms_size,NULL,&ret);
    tree_masses_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_masses_size,NULL,&ret);
    tree_tree_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_tree_size,NULL,&ret);
    
	ret = clEnqueueWriteBuffer (command_queue, mass_buffer, CL_TRUE, 0, float_data_size, (void *)masses, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, vel_buffer, CL_TRUE, 0, vector_data_size, (void *)vel, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, acc_buffer0, CL_TRUE, 0, vector_data_size, (void *)acc_old, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer (command_queue, acc_buffer1, CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, NULL);
    
    testerror(ret,"Failed to copy date from host to device: ##");
	// Create Kernel Program from source
	
    program=program_from_file("src/nbody.cl",context,device_id);
    
    // Create OpenCL Kernel
    kernel_acc = clCreateKernel(program, "compute_accelerations", &ret);
    kernel_vpos = clCreateKernel(program, "verlet_new_pos", &ret);
    kernel_vvel = clCreateKernel(program, "verlet_new_vel", &ret);
    kernel_bh = clCreateKernel(program, "barnes_hut_accelerations", &ret);
    kernel_hist=clCreateKernel(program, "count_particles", &ret);
    kernel_hist_processing=clCreateKernel(program, "process_hist", &ret);
    kernel_f2rgb=clCreateKernel(program, "f2rgb", &ret);
    testerror(ret,"Failed to create kernel ##");//add tests for other kernels

    log_timer(logt,"GPU buffers created and program compiled",0,0);

    for(int i=first_frame;i<STEPS+first_frame;i++){
        if(!(i%(STEPS<100?1:STEPS/100))){
            printf("generating frame %d/%d\n",i,STEPS+first_frame);
        }

        log_hrule(logt,'=');
        char framestr[20];
        sprintf(framestr,"FRAME %d",i);
        log_timer(logt,framestr,0,1);
        //char filename[20];
        //sprintf(filename,"ppm/F%d.ppm",i);

        //stepping position
        ret  = clSetKernelArg(kernel_vpos, 0, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_vpos, 1, sizeof (cl_mem), (void *) &vel_buffer);
        ret |= clSetKernelArg(kernel_vpos, 2, sizeof (cl_mem), (void *) acc_bufferp[i%2]);
        ret |= clSetKernelArg(kernel_vpos, 3, sizeof (cl_int), (void *) &len);
        ret |= clSetKernelArg(kernel_vpos, 4, sizeof (cl_int), (void *) &dt);
        testerror(ret,"Failed to set kernel arguments ##");

        //calculating new accelerations 
        //O(N^2)       
        ret  = clSetKernelArg(kernel_acc, 0, sizeof (cl_mem), (void *) &mass_buffer);
        ret |= clSetKernelArg(kernel_acc, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_acc, 2, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_acc, 3, sizeof (cl_int), (void *) &len);
        testerror(ret,"Failed to set kernel arguments ##");

        //BH
        //ret  = clSetKernelArg(kernel_bh, 0, sizeof (cl_mem), (void *) &acc_test_buffer);
        ret  = clSetKernelArg(kernel_bh, 0, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_bh, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_bh, 2, sizeof (cl_mem), (void *) &tree_coms_buffer);
        ret |= clSetKernelArg(kernel_bh, 3, sizeof (cl_mem), (void *) &tree_masses_buffer);
        ret |= clSetKernelArg(kernel_bh, 4, sizeof (cl_mem), (void *) &tree_tree_buffer);
        ret |= clSetKernelArg(kernel_bh, 5, sizeof (cl_int), (void *) &len);
        testerror(ret,"Failed to set kernel arguments ##");

        //stepping velocity
        ret  = clSetKernelArg(kernel_vvel, 0, sizeof (cl_mem), (void *) &vel_buffer);
        ret |= clSetKernelArg(kernel_vvel, 1, sizeof (cl_mem), (void *) acc_bufferp[i%2]);
        ret |= clSetKernelArg(kernel_vvel, 2, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_vvel, 3, sizeof (cl_int), (void *) &len);
        ret |= clSetKernelArg(kernel_vvel, 4, sizeof (cl_int), (void *) &dt);
        testerror(ret,"Failed to set kernel arguments ##");

        //histogram
        ret  = clSetKernelArg(kernel_hist, 0, sizeof (cl_mem), (void *) &hist_buffer);
        ret |= clSetKernelArg(kernel_hist, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_hist, 2, sizeof (cl_int2), (void *) &hist_dimensions);
        ret |= clSetKernelArg(kernel_hist, 3, sizeof (cl_float2), (void *) &imagebox_min);
        ret |= clSetKernelArg(kernel_hist, 4, sizeof (cl_float2), (void *) &imagebox_max);
        ret |= clSetKernelArg(kernel_hist, 5, sizeof (cl_int), (void *) &len);
        testerror(ret,"Failed to set kernel arguments ##");

        //histogram processing
        ret  = clSetKernelArg(kernel_hist_processing, 0, sizeof (cl_mem), (void *) &hist_buffer);
        ret |= clSetKernelArg(kernel_hist_processing, 1, sizeof (cl_mem), (void *) &hist_f_buffer);
        ret |= clSetKernelArg(kernel_hist_processing, 2, sizeof (cl_int2), (void *) &hist_dimensions);
        ret |= clSetKernelArg(kernel_hist_processing, 3, sizeof (cl_float2), (void *) &imagebox_min);
        ret |= clSetKernelArg(kernel_hist_processing, 4, sizeof (cl_float2), (void *) &imagebox_max);
        ret |= clSetKernelArg(kernel_hist_processing, 5, sizeof (cl_int), (void *) &len);
        //ret |= clSetKernelArg(kernel_hist_processing, 4, sizeof (cl_int), (void *) &len);
        testerror(ret,"Failed to set kernel arguments ##");

        //histogram processing
        ret  = clSetKernelArg(kernel_f2rgb, 0, sizeof (cl_mem), (void *) &hist_f_buffer);
        ret |= clSetKernelArg(kernel_f2rgb, 1, sizeof (cl_mem), (void *) &image_buffer);
        testerror(ret,"Failed to set kernel arguments ##");

        /*
        size_t global_work_size, local_work_size;  
        // Number of work items in each local work group  
        local_work_size = len;
        // Number of total work items - localSize must be devisor  
        global_work_size = (size_t) ceil( len / (float) local_work_size ) * local_work_size;
        */

        size_t global_work_size=len;

        //size_t local_work_size[2] = { 8, 8 };
        //size_t global_work_size[2] = { 1, len };
        log_partial(logt,"Kernels executions start");

        cl_event wait_event;

        ret = clEnqueueNDRangeKernel(command_queue, kernel_vpos, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Updated positions");

        ret=clEnqueueFillBuffer(command_queue,hist_buffer,&zero,sizeof(cl_int),0,hist_size,0,NULL,NULL);
        testerror(ret,"Failed to wipe hist_buffer");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_hist, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Histogram created");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_hist_processing, 2, NULL, (size_t[]){hist_width,hist_height}, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_f2rgb, 2, NULL, (size_t[]){hist_width,hist_height}, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Image created");

        //ret = clEnqueueReadBuffer(command_queue, hist_buffer, CL_TRUE, 0, hist_size, (void *)hist, 0, NULL, &wait_event);
        //testerror(ret,"Failed to copy data from device to host ##");
        //ret = clEnqueueReadBuffer(command_queue, hist_f_buffer, CL_TRUE, 0, hist_f_size, (void *)hist_f, 0, NULL, &wait_event);
        //testerror(ret,"Failed to copy data from device to host ##");

        ret = clEnqueueReadBuffer(command_queue, image_buffer, CL_TRUE, 0, image_size, (void *)image, 0, NULL, &wait_event);
        testerror(ret,"Failed to copy data from device to host ##");


        ret = clEnqueueReadBuffer(command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, &wait_event);
        testerror(ret,"Failed to copy data from device to host ##");
        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Copied positions and image buffers to host");

#ifdef BH

        oct_tree *testtree=newtree(pos,tree_masses,tree_coms,tree_tree,TREE_SIZE);
        //add_leaf(testtree,0,pos[0]);
        for(int j=0;j<len;j++){
            add_leaf(testtree,j,pos[j]);
        }
        //printf("%d\n",testtree->nodes_num);

        log_partial(logt,"Octree built");

        compute_masses(testtree,masses,pos);

        log_partial(logt,"Nodes COMs computed");

        //ret = clEnqueueReadBuffer(command_queue, *acc_bufferp[(i+1)%2], CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, &wait_event);
        
        ret  = clEnqueueWriteBuffer (command_queue, tree_coms_buffer, CL_TRUE, 0, tree_coms_size, (void *)tree_coms, 0, NULL, NULL);
        ret |= clEnqueueWriteBuffer (command_queue, tree_masses_buffer, CL_TRUE, 0, tree_masses_size, (void *)tree_masses, 0, NULL, NULL);
        ret |= clEnqueueWriteBuffer (command_queue, tree_tree_buffer, CL_TRUE, 0, tree_tree_size, (void *)tree_tree, 0, NULL, NULL);
        testerror(ret,"Failed to copy tree buffers");

        free(testtree);

        ret = clEnqueueNDRangeKernel(command_queue, kernel_bh, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");
#else
        ret = clEnqueueNDRangeKernel(command_queue, kernel_acc, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");
#endif

        /*
        fptr=secure_fopen("tempdata.csv","w");
        dump_positions(fptr,pos,color,len);
        fclose(fptr);
        log_partial(logt,"Saved positions file");
        */


        //char filename[20];
        //sprintf(filename,"data/F%d.bin",i);
#if 0
        fptr=secure_fopen("tempdata_hist.bin","wb");
        for(int xx=0;xx<hist_width;xx++){
            float x=xx*(imagebox_max.x-imagebox_min.x)/hist_width+XMIN;
            for(int yy=0;yy<hist_height;yy++){
                float y=yy*(imagebox_max.y-imagebox_min.y)/hist_height+YMIN;
                //float brightness=(float)hist[hist_width*xx+yy]/len*hist_width*hist_height/(XMAX-XMIN)/(YMAX-YMIN);
                float brightness=hist_f[hist_height*yy+xx];
                fwrite(&x,sizeof(float),1,fptr);
                fwrite(&y,sizeof(float),1,fptr);
                fwrite(&brightness,sizeof(float),1,fptr);
                //fprintf(fptr,"%f,%f,%f\n",x,y,brightness);
            }
        }
        fclose(fptr);
        log_partial(logt,"Saved hist file");

        char cmdstr[40];
        /*
        sprintf(cmdstr,"gnuplot -c nbody2.plt %d",i);
        system(cmdstr);
        */
        sprintf(cmdstr,"gnuplot -c nbody2_hist.plt %d",i);
        system(cmdstr);
#endif

        char ppm_cmd[40];
        sprintf(ppm_cmd,"convert temp.ppm frames/F%d.png",i);
        fptr=fopen("temp.ppm","wb");
        fprintf(fptr,"P6\n%i %i 255\n",HIST_WIDTH,HIST_HEIGHT);
        fwrite(image,sizeof(char),image_size,fptr);
        fclose(fptr);
        system(ppm_cmd);

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Accelerations computed (and image output)");


        ret = clEnqueueNDRangeKernel(command_queue, kernel_vvel, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        testerror(ret,"Failed to execute kernel for execution ##");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Updated velocities");

        /*
        ret = clEnqueueReadBuffer(command_queue, acc_test_buffer, CL_TRUE, 0, vector_data_size, (void *)acc_test, 0, NULL, &wait_event);
        ret|= clEnqueueReadBuffer(command_queue, *acc_bufferp[(i+1)%2], CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, &wait_event);
        testerror(ret,"Failed to copy accelerations from device to host");

        for(int j=0;j<20;j++){
            printf("%10f %10f %.2f\n",acc[j].x,acc_test[j].x,2*fabs((acc[j].x-acc_test[j].x)/(acc[j].x+acc_test[j].x)));
        }
        */
        
    }

    log_hrule(logt,'=');

    ret =clEnqueueReadBuffer(command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, vel_buffer, CL_TRUE, 0, vector_data_size, (void *)vel, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, *acc_bufferp[(STEPS+1)%2], CL_TRUE, 0, vector_data_size, (void *)acc_old, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, *acc_bufferp[STEPS%2], CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, NULL);
    testerror(ret,"Failed to copy data from device to host ##");

    fptr=secure_fopen("final_state.csv","w");
    dump_state(fptr,color,masses,pos,vel,acc_old,acc,len);
    fclose(fptr);

    log_timer(logt,"Final state dumped",0,0);

	/* Display Result */
    

	/* Finalization */

    /* free device resources */
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel_acc);
    clReleaseKernel(kernel_vpos);
    clReleaseKernel(kernel_vvel);
	clReleaseProgram(program);

	clReleaseMemObject(mass_buffer);
    clReleaseMemObject(pos_buffer);
    clReleaseMemObject(vel_buffer);
    clReleaseMemObject(acc_buffer0);
    clReleaseMemObject(acc_buffer1);

    clReleaseMemObject(acc_test_buffer);
    clReleaseMemObject(tree_coms_buffer);
    clReleaseMemObject(tree_masses_buffer);
    clReleaseMemObject(tree_tree_buffer);

    clReleaseMemObject(hist_buffer);
    clReleaseMemObject(hist_f_buffer);
    clReleaseMemObject(image_buffer);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

    /* free host resources */
	//free(source_str);

    log_timer(logt,"Resources freed",0,0);

    fclose(logptr);

    return 0;
}