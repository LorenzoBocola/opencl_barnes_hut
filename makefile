OUT = main
CC = gcc
ODIR = obj
SDIR = src

_OBJS = clutils.o io.o bh.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

CFLAGS = -O3 -march=native -lm -lOpenCL

$(OUT): $(SDIR)/main.c $(OBJS)
	$(CC) $(SDIR)/main.c $(OBJS) $(CFLAGS) -o $(OUT)

-include $(OBJS:.o=.d)

$(ODIR)/%.o: $(SDIR)/%.c
	gcc -MMD -c $(CFLAGS) $(SDIR)/$*.c -o $@

.PHONY: video clean prep

video:
#	ffmpeg -framerate 60 -i frames/F%d.png  -c:v libx264 -pix_fmt yuv420p  -profile:v baseline -level 3.0  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vb 1024k  -acodec aac -ar 44100 -ac 2 -minrate 1024k -maxrate 1024k -bufsize 1024k  -movflags +faststart  nbody.mp4 -y
	ffmpeg -framerate 60 -i frames/F%d.png nbody.mp4 -y

clean:
	rm -f temp.ppm
	rm -f frames/*

prep:
	mkdir obj
	mkdir frames