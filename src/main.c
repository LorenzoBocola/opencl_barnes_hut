#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "clutils.h"
#include "io.h"

#include "bh.h"

#ifndef M_PI//vscode shenanigans
#define M_PI 0
//#error "M_PI undefined"
#endif

#define PARTICLES 100000
#define STEPS 1000
#define DT 0.005

#define XMIN -10.
#define XMAX 10.
#define YMIN -10.
#define YMAX 10.
#define ZMIN -10.
#define ZMAX 10.

#define TREE_SIZE (3*PARTICLES)

#define BH

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGTH 1080

#define INIT_TYPE 0


void *secure_malloc(size_t size){
    void *ptr=malloc(size);
    if(ptr==NULL){
        fprintf(stderr,"Error: failed to allocate %ld bytes of memory\n",size);
        exit(1);
    }
    return ptr;
}

void init_state(cl_float *masses,float *color,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int particles){
    for(int i=0;i<particles;i++){
        /*
            the position and velocity distributions were chosen to produce a
            visually interesting output and aren't by any means physically
            accurate
        */
        acc[i]=(cl_float3){0,0,0};
        acc_old[i]=(cl_float3){0,0,0};
#if INIT_TYPE==0
        masses[i]=2000./particles;
        if(i<particles/4){
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
#elif INIT_TYPE==1
        masses[i]=2000./particles;
        if(i<particles/4){
            color[i]=0;
            float r=0.5*drand48();
            float theta=drand48()*2*M_PI;
            pos[i]=(cl_float3){r*cos(theta)+3,0.75,r*sin(theta),0.05*(2*drand48()-1)};
            vel[i]=(cl_float3){-sqrt(r)*sin(theta)-1,0,sqrt(r)*cos(theta)};
        }else{
            color[i]=1;
            float r=drand48();
            float theta=drand48()*2*M_PI;
            pos[i]=(cl_float3){r*cos(theta)-1,r*sin(theta),0.1*(2*drand48()-1)};
            vel[i]=(cl_float3){-1.5*sqrt(r)*sin(theta)+0.333,1.5*sqrt(r)*cos(theta),0};
        }
#elif INIT_TYPE==2
        masses[i]=3000./particles;
        float r=drand48();
        float theta=drand48()*2*M_PI;
        pos[i]=(cl_float3){r*cos(theta),r*sin(theta),0.1*(2*drand48()-1)};
        vel[i]=(cl_float3){-1.5*sqrt(r)*sin(theta),1.5*sqrt(r)*cos(theta),0};
        if(i<particles/2){
            color[i]=0;
            pos[i].x-=2;
        }else{
            color[i]=1;
            pos[i].x+=2;
        }
#else
#error invalid INIT_TYPE value
#endif
    }
}

int main(int argc, char *argv[]){
    srand48(time(NULL));

    FILE *logptr=secure_fopen("nbody.log","w");
    logtimer *logt=create_timer(logptr);
    //log_partial(logt,"start");

    cl_int particles;
    int first_frame=0;
    if(argc<2){
        particles=PARTICLES;
    }else{
        if(argc<4){
            fprintf(stderr,"Error: invalid number of arguments\n");
            exit(1);
        }
        particles=atoi(argv[2]);
        first_frame=atoi(argv[3]);
    }

    cl_float dt=DT;

    cl_int image_width=IMAGE_WIDTH;
    cl_int image_height=IMAGE_HEIGTH;

    cl_float2 imagebox_min={-4,-4*0.5625};
    cl_float2 imagebox_max={4,4*0.5625};

    cl_float2 xrange={XMIN,XMAX};
    cl_float2 yrange={YMIN,YMAX};
    cl_float2 zrange={ZMIN,ZMAX};
    ot_bbox bbox={.xrange=xrange,.yrange=yrange,.zrange=zrange};

    cl_int2 image_dimensions={image_width,image_height};

    //allocating host resources

    FILE *fptr;

    size_t float_data_size=particles*sizeof(cl_float);
    size_t vector_data_size=particles*sizeof(cl_float3);

    float *color=secure_malloc(particles*sizeof(float));//no longer used
    cl_float *masses=secure_malloc(particles*sizeof(cl_float));
    cl_float3 *pos=secure_malloc(particles*sizeof(cl_float3));
    cl_float3 *vel=secure_malloc(particles*sizeof(cl_float3));
    cl_float3 *acc_old=secure_malloc(particles*sizeof(cl_float3));
    cl_float3 *acc=secure_malloc(particles*sizeof(cl_float3));

    size_t tree_coms_size=TREE_SIZE*sizeof(cl_float3);
    size_t tree_masses_size=TREE_SIZE*sizeof(cl_float);
    size_t tree_children_size=8*TREE_SIZE*sizeof(cl_int);
    size_t tree_leaves_size=TREE_SIZE*sizeof(cl_int);

    size_t particles_count_size=image_width*image_height*sizeof(cl_int);
    size_t image_f_size=image_width*image_height*sizeof(cl_float);
    size_t image_size=image_width*image_height*3*sizeof(cl_char);

    cl_float3 *tree_coms=secure_malloc(TREE_SIZE*sizeof(cl_float3));
    cl_float *tree_masses=secure_malloc(TREE_SIZE*sizeof(cl_float));
    cl_int *tree_children=secure_malloc(8*TREE_SIZE*sizeof(cl_int));
    cl_int *tree_leaves=secure_malloc(TREE_SIZE*sizeof(cl_int));

    //cl_int *particles_count=secure_malloc(particles_count_size);
    //cl_float *image_f=secure_malloc(image_f_size);
    cl_char *image=secure_malloc(image_size);

    log_timer(logt,"Host memory allocated",0,0);
    
    //initializing data
    if(argc<2){
        //generating data
        init_state(masses,color,pos,vel,acc_old,acc,particles);
    }else{
        //loading data from state file
        fptr=secure_fopen(argv[1],"r");
        load_state(fptr,color,masses,pos,vel,acc_old,acc,particles);
        fclose(fptr);
    }

    //allocating OpenCL device resources
	cl_int ret;

    //creating context
    cl_context context = NULL;
	cl_device_id device_id = NULL;
    create_context(&context,&device_id);

    //creating command queue
    cl_command_queue command_queue=clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    testerror(ret,"Failed to create command queue");

    //creating buffers

    //used for the physical calculations
    cl_mem mass_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, float_data_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem pos_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem vel_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem acc_buffer0 = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem acc_buffer1 = clCreateBuffer (context, CL_MEM_READ_WRITE, vector_data_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");

    //BH stuff
    cl_mem tree_coms_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_coms_size,NULL,&ret);
    testerror(ret,"Failed to create buffer");
    cl_mem tree_masses_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_masses_size,NULL,&ret);
    testerror(ret,"Failed to create buffer");
    cl_mem tree_children_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_children_size,NULL,&ret);
    testerror(ret,"Failed to create buffer");
    cl_mem tree_leaves_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,tree_leaves_size,NULL,&ret);
    testerror(ret,"Failed to create buffer");

    /*
        used to alternate the old and new accelerations buffers without having
        to copy any data
    */
    cl_mem *acc_bufferp[2]={&acc_buffer0,&acc_buffer1};

    //used to output an image
    cl_mem particles_count_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, particles_count_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem image_f_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, image_f_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");
    cl_mem image_RGB_buffer=clCreateBuffer (context, CL_MEM_READ_WRITE, image_size, NULL, &ret);
    testerror(ret,"Failed to create buffer");

    //writing to the buffers
    ret = clEnqueueWriteBuffer (command_queue, mass_buffer, CL_TRUE, 0, float_data_size, (void *)masses, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, vel_buffer, CL_TRUE, 0, vector_data_size, (void *)vel, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer (command_queue, acc_buffer0, CL_TRUE, 0, vector_data_size, (void *)acc_old, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer (command_queue, acc_buffer1, CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, NULL);
    testerror(ret,"Failed to copy data from host to device");

    //creating program from .cl file
    cl_program program=program_from_file("src/nbody.cl",context,device_id);
    
    //creating OpenCL kernels
    cl_kernel kernel_acc = clCreateKernel(program, "compute_accelerations", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_vpos = clCreateKernel(program, "verlet_new_pos", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_vvel = clCreateKernel(program, "verlet_new_vel", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_bh = clCreateKernel(program, "barnes_hut_accelerations", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_count_particles=clCreateKernel(program, "count_particles", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_image_processing=clCreateKernel(program, "process_image", &ret);
    testerror(ret,"Failed to create kernel");
    cl_kernel kernel_f2rgb=clCreateKernel(program, "f2rgb", &ret);
    testerror(ret,"Failed to create kernel");

    log_timer(logt,"OpenCL device resourced allocated",0,0);

    for(int i=first_frame;i<STEPS+first_frame;i++){
        //simulation loop

        if(!(i%(STEPS<100?1:STEPS/100))){
            printf("generating frame %d/%d\n",i,STEPS+first_frame);
        }
        log_hrule(logt,'=');
        char framestr[20];
        sprintf(framestr,"FRAME %d",i);
        log_timer(logt,framestr,0,1);

        //setting kernel arguments

        //stepping position kernel
        ret  = clSetKernelArg(kernel_vpos, 0, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_vpos, 1, sizeof (cl_mem), (void *) &vel_buffer);
        ret |= clSetKernelArg(kernel_vpos, 2, sizeof (cl_mem), (void *) acc_bufferp[i%2]);
        ret |= clSetKernelArg(kernel_vpos, 3, sizeof (cl_int), (void *) &particles);
        ret |= clSetKernelArg(kernel_vpos, 4, sizeof (cl_int), (void *) &dt);
        testerror(ret,"Failed to set positions step kernel arguments");

        //O(N^2) accelerations calculating kernel      
        ret  = clSetKernelArg(kernel_acc, 0, sizeof (cl_mem), (void *) &mass_buffer);
        ret |= clSetKernelArg(kernel_acc, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_acc, 2, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_acc, 3, sizeof (cl_int), (void *) &particles);
        testerror(ret,"Failed to set O(N^2) accelerations calculation kernel arguments");

        //BH kernel
        ret  = clSetKernelArg(kernel_bh, 0, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_bh, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_bh, 2, sizeof (cl_mem), (void *) &tree_coms_buffer);
        ret |= clSetKernelArg(kernel_bh, 3, sizeof (cl_mem), (void *) &tree_masses_buffer);
        ret |= clSetKernelArg(kernel_bh, 4, sizeof (cl_mem), (void *) &tree_children_buffer);
        ret |= clSetKernelArg(kernel_bh, 5, sizeof (cl_mem), (void *) &tree_leaves_buffer);
        ret |= clSetKernelArg(kernel_bh, 6, sizeof (cl_int), (void *) &particles);
        ret |= clSetKernelArg(kernel_bh, 7, sizeof (cl_float), (void *) &(cl_float){xrange.s1-xrange.s0});
        testerror(ret,"Failed to set BH accelerations calculation kernel arguments");

        //velocities stepping kernel
        ret  = clSetKernelArg(kernel_vvel, 0, sizeof (cl_mem), (void *) &vel_buffer);
        ret |= clSetKernelArg(kernel_vvel, 1, sizeof (cl_mem), (void *) acc_bufferp[i%2]);
        ret |= clSetKernelArg(kernel_vvel, 2, sizeof (cl_mem), (void *) acc_bufferp[(i+1)%2]);
        ret |= clSetKernelArg(kernel_vvel, 3, sizeof (cl_int), (void *) &particles);
        ret |= clSetKernelArg(kernel_vvel, 4, sizeof (cl_int), (void *) &dt);
        testerror(ret,"Failed to set velocities step kernel arguments");

        //counting particles in each pixel kernel
        ret  = clSetKernelArg(kernel_count_particles, 0, sizeof (cl_mem), (void *) &particles_count_buffer);
        ret |= clSetKernelArg(kernel_count_particles, 1, sizeof (cl_mem), (void *) &pos_buffer);
        ret |= clSetKernelArg(kernel_count_particles, 2, sizeof (cl_int2), (void *) &image_dimensions);
        ret |= clSetKernelArg(kernel_count_particles, 3, sizeof (cl_float2), (void *) &imagebox_min);
        ret |= clSetKernelArg(kernel_count_particles, 4, sizeof (cl_float2), (void *) &imagebox_max);
        ret |= clSetKernelArg(kernel_count_particles, 5, sizeof (cl_int), (void *) &particles);
        testerror(ret,"Failed to set particles counting kernel arguments");

        //image processing kernel
        ret  = clSetKernelArg(kernel_image_processing, 0, sizeof (cl_mem), (void *) &particles_count_buffer);
        ret |= clSetKernelArg(kernel_image_processing, 1, sizeof (cl_mem), (void *) &image_f_buffer);
        ret |= clSetKernelArg(kernel_image_processing, 2, sizeof (cl_int2), (void *) &image_dimensions);
        ret |= clSetKernelArg(kernel_image_processing, 3, sizeof (cl_float2), (void *) &imagebox_min);
        ret |= clSetKernelArg(kernel_image_processing, 4, sizeof (cl_float2), (void *) &imagebox_max);
        ret |= clSetKernelArg(kernel_image_processing, 5, sizeof (cl_int), (void *) &particles);
        testerror(ret,"Failed to set image processing kernel arguments");

        //image conversion (from [0,1] float to RGB) kernel
        ret  = clSetKernelArg(kernel_f2rgb, 0, sizeof (cl_mem), (void *) &image_f_buffer);
        ret |= clSetKernelArg(kernel_f2rgb, 1, sizeof (cl_mem), (void *) &image_RGB_buffer);
        testerror(ret,"Failed to set image conversion kernel arguments");

        size_t global_work_size=particles;

        log_partial(logt,"Kernel arguments set");

        cl_event wait_event;

        //updating the positions
        ret = clEnqueueNDRangeKernel(command_queue, kernel_vpos, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute positions step kernel");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Positions updated");

        //generating the image
        ret=clEnqueueFillBuffer(command_queue,particles_count_buffer,&(cl_int){0},sizeof(cl_int),0,particles_count_size,0,NULL,NULL);
        testerror(ret,"Failed to wipe particles_count_buffer");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_count_particles, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute particles counting kernel");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_image_processing, 2, NULL, (size_t[]){image_width,image_height}, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute image processing kernel");

        ret = clEnqueueNDRangeKernel(command_queue, kernel_f2rgb, 2, NULL, (size_t[]){image_width,image_height}, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute image conversion kernel");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Image created");

        ret = clEnqueueReadBuffer(command_queue, image_RGB_buffer, CL_TRUE, 0, image_size, (void *)image, 0, NULL, &wait_event);
        testerror(ret,"Failed to copy buffer image_RGB_buffer from device to host");

        ret = clEnqueueReadBuffer(command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, &wait_event);
        testerror(ret,"Failed to copy buffer pos_buffer from device to host");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Positions and image buffers copied to host");

        //computing the new accelerations
#ifdef BH
        oct_tree *tree=newtree(pos,tree_masses,tree_coms,tree_children,tree_leaves,TREE_SIZE,bbox);
        
        fill_tree(tree,pos,particles);
        log_partial(logt,"Octree built");

        summarize_tree(tree,masses,pos);
        log_partial(logt,"Tree summarized");

        ret  = clEnqueueWriteBuffer (command_queue, tree_coms_buffer, CL_TRUE, 0, tree_coms_size, (void *)tree_coms, 0, NULL, NULL);
        ret |= clEnqueueWriteBuffer (command_queue, tree_masses_buffer, CL_TRUE, 0, tree_masses_size, (void *)tree_masses, 0, NULL, NULL);
        ret |= clEnqueueWriteBuffer (command_queue, tree_children_buffer, CL_TRUE, 0, tree_children_size, (void *)tree_children, 0, NULL, NULL);
        ret |= clEnqueueWriteBuffer (command_queue, tree_leaves_buffer, CL_TRUE, 0, tree_leaves_size, (void *)tree_leaves, 0, NULL, NULL);
        testerror(ret,"Failed to copy tree buffers");

        free(tree);

        ret = clEnqueueNDRangeKernel(command_queue, kernel_bh, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute BH accelerations computing kernel");
#else
        ret = clEnqueueNDRangeKernel(command_queue, kernel_acc, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute O(N^2) accelerations computing kernel");
#endif

        //converting the image from .ppm to .png (while the accelerations are being computed)
        char ppm_cmd[40];
        sprintf(ppm_cmd,"convert temp.ppm frames/F%d.png",i);
        fptr=secure_fopen("temp.ppm","wb");
        fprintf(fptr,"P6\n%i %i 255\n",IMAGE_WIDTH,IMAGE_HEIGTH);
        fwrite(image,sizeof(char),image_size,fptr);
        fclose(fptr);
        int ppm_cmdrv=system(ppm_cmd);
        if(ppm_cmdrv){
            fprintf(stderr,"Warning: imagemagick convert returned code %d\n",ppm_cmdrv);
        }

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Accelerations computed (and image output)");

        //computing the velocities
        ret = clEnqueueNDRangeKernel(command_queue, kernel_vvel, 1, NULL, &global_work_size, NULL, 0, NULL, &wait_event);
        testerror(ret,"Failed to execute velocities step kernel");

        clWaitForEvents(1,&wait_event);
        log_partial(logt,"Velocities updated");
    }

    log_hrule(logt,'=');

    //dumping the simulation's state
    ret =clEnqueueReadBuffer(command_queue, pos_buffer, CL_TRUE, 0, vector_data_size, (void *)pos, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, vel_buffer, CL_TRUE, 0, vector_data_size, (void *)vel, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, *acc_bufferp[(STEPS+1)%2], CL_TRUE, 0, vector_data_size, (void *)acc_old, 0, NULL, NULL);
    ret|=clEnqueueReadBuffer(command_queue, *acc_bufferp[STEPS%2], CL_TRUE, 0, vector_data_size, (void *)acc, 0, NULL, NULL);
    testerror(ret,"Failed to copy data from device to host");

    fptr=secure_fopen("final_state.csv","w");
    dump_state(fptr,color,masses,pos,vel,acc_old,acc,particles);
    fclose(fptr);

    log_timer(logt,"Final state dumped",0,0);

    //freeing resources
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

    clReleaseMemObject(tree_coms_buffer);
    clReleaseMemObject(tree_masses_buffer);
    clReleaseMemObject(tree_children_buffer);
    clReleaseMemObject(tree_leaves_buffer);

    clReleaseMemObject(particles_count_buffer);
    clReleaseMemObject(image_f_buffer);
    clReleaseMemObject(image_RGB_buffer);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

    log_timer(logt,"Resources freed",0,0);

    fclose(logptr);

    return 0;
}