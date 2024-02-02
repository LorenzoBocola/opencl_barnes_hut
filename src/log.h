#ifndef _LOG_H_
#define _LOG_H_

#include <stdio.h>

typedef struct _log_timer_struct logtimer;

logtimer *create_timer(FILE *fptr);
void log_timer(logtimer *timer,const char *str,int partial,int update);
#define log_partial(timer,str) log_timer(timer,str,1,1)
void log_hrule(logtimer *timer,char ch);

#endif