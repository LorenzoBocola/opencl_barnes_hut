#ifndef _IO_H_
#define _IO_H_

#include <stdio.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

//data
FILE *secure_fopen(const char *filename, const char *mode);
void dump_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len);
void load_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len);

//log
typedef struct _log_timer_struct logtimer;

logtimer *create_timer(FILE *fptr);
void log_timer(logtimer *timer,const char *str,int partial,int update);
#define log_partial(timer,str) log_timer(timer,str,1,1)
void log_hrule(logtimer *timer,char ch);

#endif