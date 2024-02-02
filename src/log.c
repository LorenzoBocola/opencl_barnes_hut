#include "log.h"

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

#define LOG_LINE_LEN 80

void malloc_check(void *ptr){
    if(ptr==NULL){
        fprintf(stderr,"Error: could not allocate memory\n");
        exit(1);
    }
}

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
    malloc_check(timer);

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