Halide Renderscript "blur" and "copy" samples
===

These samples demonstrates two supported modes of operation for Renderscript codegen:
 - vectorized operations over interleavead RGBA images;
 - single-element operations over planar images.

Prerequisites
---

Connected Android device needs to be "rooted", following initial set up needs to be done on the device. You will need Android sdk in the path:
```
adb shell
su
mkdir /data/tmp
mount -o remount,rw /system
```

Building and running
---

```
$ make
...
Planar blur:
Ran 500 reps. One rep times:
RS:  19.504922ms
ARM: 3.329536ms
...
Planar copy:
Ran 500 reps. One rep times:
RS:  6.202582ms
ARM: 1.496806ms
...
Interleaved(vectorized) blur:
Ran 500 reps. One rep times:
RS:  2.609520ms
ARM: 1.559394ms
...
Interleaved(vectorized) copy:
Ran 500 reps. One rep times:
RS:  2.192948ms
ARM: 0.421188ms
```
