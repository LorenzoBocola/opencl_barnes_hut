#ifndef _CLUTILS_H_
#define _CLUTILS_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define testerror(ret,msg) testerrorfunc(ret,msg,__FILE__,__LINE__)

void testerrorfunc(cl_int ret,const char *msg,const char *file,int line);

cl_int create_context(cl_context *context,cl_device_id *device_id);
cl_program program_from_file(const char *filename,cl_context context,cl_device_id device_id);

#endif