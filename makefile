OUT = main
CC = gcc
ODIR = obj
SDIR = src

_OBJS = clutils.o log.o bh.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

CFLAGS = -O3 -march=native -lm -lOpenCL #-Wall

DBOUT = main_db
DBDIR = dbg
DBCFLAGS = -g -O0

$(OUT): $(SDIR)/main.c $(OBJS)
	$(CC) $(SDIR)/main.c $(OBJS) $(CFLAGS) -o $(OUT)

#$(ODIR)/math_parsing.o: $(patsubst %,$(SDIR)/%,math_parsing.c math_parsing.h)
#	$(CC) -MMD -c -o $@ $<
#
#$(ODIR)/math_wrapper.o: $(patsubst %,$(SDIR)/%,math_wrapper.c math_wrapper.h)
#	$(CC) -MMD -c -o $@ $<
#
#$(ODIR)/%.o: $(SDIR)/%.c 
#    $(CC) -c $(INC) -o $@ $< $(CFLAGS) 

# pull in dependency info for *existing* .o files
-include $(OBJS:.o=.d)

# compile and generate dependency info
$(ODIR)/%.o: $(SDIR)/%.c
	gcc -MMD -c $(CFLAGS) $(SDIR)/$*.c -o $@
#	gcc -MM $(CFLAGS) $*.c > $*.d

.PHONY: video clean

video:
#	ffmpeg -framerate 60 -i frames/F%d.png  -c:v libx264 -pix_fmt yuv420p  -profile:v baseline -level 3.0  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vb 1024k  -acodec aac -ar 44100 -ac 2 -minrate 1024k -maxrate 1024k -bufsize 1024k  -movflags +faststart  nbody.mp4 -y
	ffmpeg -framerate 60 -i frames/F%d.png nbody.mp4 -y

clean:
	rm -f data/*
	rm -f frames/*