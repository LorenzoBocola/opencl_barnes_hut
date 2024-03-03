#ifndef _CLUTILS_H_
#define _CLUTILS_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define check_cl_ret(ret,format,...) check_cl_ret_(__FILE__,__LINE__,ret,format __VA_OPT__(,) __VA_ARGS__)

void check_cl_ret_(const char *file, int line,cl_int ret,const char *format,...);

cl_int create_context(cl_context *context,cl_device_id *device_id);
cl_program program_from_file(const char *filename,cl_context context,cl_device_id device_id);

#endif//_CLUTILS_H_