#include "clutils.h"

#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE 1000000

void testerrorfunc(cl_int ret,const char *msg,const char *file,int line){
    if(ret!=CL_SUCCESS){
        fprintf(stderr,"Error in %s at line %d: %d\n",file,line,ret);
        fprintf(stderr,"%s\n",msg);
        exit(1);
    }
}

size_t loadtext(const char *filename,char *dest){
    FILE *fptr=fopen(filename,"r");
    size_t source_size=fread(dest,sizeof(char),MAX_SOURCE_SIZE,fptr);
    fclose(fptr);
    return source_size;
}

cl_int create_context(cl_context *context,cl_device_id *device_id){
    cl_int ret;
    cl_uint ret_num_platforms;
    cl_platform_id platform_id;
    cl_device_id device_id_tmp;
    // Platform
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	testerror(ret,"Failed to get platform ID.");
	// Device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id_tmp, NULL);
    testerror(ret,"Failed to get device ID.");
	// Context
	cl_context context_tmp = clCreateContext(NULL, 1, &device_id_tmp, NULL, NULL, NULL);//&ret);
    testerror(ret,"Failed to create OpenCL context.");
    *context=context_tmp;
    *device_id=device_id_tmp;
    return 0;
}

cl_program program_from_file(const char *filename,cl_context context,cl_device_id device_id){
    cl_int ret;

    char *source_str=malloc(MAX_SOURCE_SIZE*sizeof(char));
    size_t source_size=loadtext(filename,source_str);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
			(const size_t *)&source_size, &ret);
	testerror(ret,"Failed to create OpenCL program from source ##");
	// Build Kernel Program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
		printf ("Error in kernel: %s\n", build_log);
		exit(1);
	}

    free(source_str);

    return program;
}