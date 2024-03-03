#include "io.h"

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

//data
#define PRINT_VECTOR(fptr,s,vec) fprintf(fptr,"%f,%f,%f"s,(vec).x,(vec).y,(vec).z);
#define SCAN_VECTOR(fptr,s,vecp) fscanf(fptr,"%f,%f,%f"s,&((vecp)->x),&((vecp)->y),&((vecp)->z));

FILE *secure_fopen(const char *filename, const char *mode){
    FILE *fptr=fopen(filename,mode);
    if(fptr==NULL){
        fprintf(stderr,"Error: file \"%s\" could not be opened\n",filename);
        exit(1);
    }
    return fptr;
}

void dump_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,
    cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len){
    for(int i=0;i<len;i++){
        fprintf(fptr,"%f,",color[i]);
        fprintf(fptr,"%f,",masses[i]);
        PRINT_VECTOR(fptr,",",pos[i]);
        PRINT_VECTOR(fptr,",",vel[i]);
        PRINT_VECTOR(fptr,",",acc_old[i]);
        PRINT_VECTOR(fptr,"\n",acc[i]);
    }
}

void load_state(FILE *fptr,float *color,cl_float *masses,cl_float3 *pos,
    cl_float3 *vel,cl_float3 *acc_old,cl_float3 *acc,cl_int len){
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

void dump_positions(FILE *fptr,cl_float3 *pos,float *color,cl_int len){
    for(int i=0;i<len;i++){
        PRINT_VECTOR(fptr,",",pos[i]);
        fprintf(fptr,"%f\n",color[i]);
    }
}

//log
#define LOG_LINE_LEN 80

struct _log_timer_struct{
    FILE *fptr;
    struct timespec start;
    struct timespec partial;
};

long long int timediff_ns(struct timespec start, struct timespec finish){
    return (finish.tv_sec-start.tv_sec)*1000000000+finish.tv_nsec-start.tv_nsec;
}

logtimer *create_timer(FILE *fptr){
    logtimer *timer=malloc(sizeof(logtimer));
    if(timer==NULL){
        fprintf(stderr,"Error: could not allocate memory\n");
        exit(1);
    }

    timer->fptr=fptr;
    struct timespec time;
    clock_gettime(CLOCK_REALTIME,&time);
    timer->start=time;
    timer->partial=time;

    return timer;
}

void log_timer(logtimer *timer,const char *str,int partial,int update){
    struct timespec time;
    clock_gettime(CLOCK_REALTIME,&time);
    long long int diff=timediff_ns(partial?timer->partial:timer->start,time);
    if(update){
        timer->partial=time;
    }
    char timestr[LOG_LINE_LEN];
    sprintf(timestr,"%s%f ms",partial?"+":"",(float)diff/1000000);
    fprintf(timer->fptr,"%s",str);
    for(int i=0;i<LOG_LINE_LEN-strlen(str)-strlen(timestr);i++){
        fputc('.',timer->fptr);
    }
    fprintf(timer->fptr,"%s\n",timestr);
}

void log_hrule(logtimer *timer,char ch){
    for(int i=0;i<LOG_LINE_LEN;i++){
        fputc(ch,timer->fptr);
    }
    fputc('\n',timer->fptr);
}