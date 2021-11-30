import os
import sys

# Run in MS-DOS: hypoell-loc.py hypoel

def checkRequiredFiles(f):
    if not os.path.exists(f):
        print("{0} is not found!".format(f))
        sys.exit()

if len(sys.argv) != 2:
    print("\n+++ usage: hypoell-loc.py inputName\n")
    sys.exit()

root = sys.argv[1]

# PHASE FILE 
filepick="{0}.pha".format(root)
checkRequiredFiles(filepick)

# VELOCITY FILE
filevel="{0}.prm".format(root)
checkRequiredFiles(filevel)

# STATION FILE
filesta="{0}.sta".format(root)
checkRequiredFiles(filesta)

# $root.out : FINAL LOCATION FILE
# $root.sum : SUMMARY FILE
# $root.arc : POLARITIES

# PARAMETER FILE
filepar="default.cfg"
checkRequiredFiles(filepar)

# REFERNCE TIME
date="19800101"

# CONTROL FILE
filecom="{0}.ctl".format(root)
with open(filecom, "w") as f:
    f.write("stdin\n")
    f.write("y\n")
    f.write("{0}\n".format(root))
    f.write("{0}.log\n".format(root))
    f.write("{0}.out\n".format(root))
    f.write("y\n")
    f.write("{0}.sum\n".format(root))
    f.write("y\n")
    f.write("{0}.arc\n".format(root))
    f.write("n\n")
    f.write("n\n")
    f.write("jump {0}\n".format(filepar))
    f.write("jump {0}\n".format(filevel))
    f.write("begin station list +1 {0}\n".format(date))
    f.write("jump {0}\n".format(filesta))
    f.write("arrival times next\n")
    f.write("jump {}\n".format(filepick))

with open("{0}_runHE.bat".format(root), "w") as f:
    f.write("hymain.exe < {0} > NUL\n ".format(filecom))

# Run Hypoellipse
cmd = "{0}_runHE.bat > NUL".format(root)
os.system(cmd)

for f in ["{0}.1st".format(root),
          "{0}.2st".format(root),
          "{0}.3sc".format(root),
          "{0}.4sc".format(root),
          "{0}.5sc".format(root),
          "{0}.arc".format(root),
          "{0}.log".format(root),
          "{0}.sum".format(root),
          "{0}_runHE.bat".format(root),
          filecom]:
    if os.path.exists(f):
        os.remove(f)