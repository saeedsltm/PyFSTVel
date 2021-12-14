import os, sys, platform

# Check required files
def checkRequiredFiles(f):
    if not os.path.exists(f):
        print("{0} is not found!".format(f))
        sys.exit()

# Get mean of a list
def mean(a):
    return sum(a)/len(a)

# Get statistics of hypo71 outputs
def getStatistic(root):
    RMS, ERH, ERZ = [], [], []
    with open("{0}.out".format(root)) as f:
        for l in f:
            if "date    origin" in l:
                l = next(f)
                rms = float(l[65:72])
            if "seh  sez" in l:
                l = next(f)
                erh = float(l[2:7])
                erz = float(l[7:12])
                RMS.append(rms)
                ERH.append(erh)
                ERZ.append(erz)
    return mean(RMS), mean(ERH), mean(ERZ)

# Run Hypoellipse
def runHypoellipse(inputName):
    # Define rootName
    root = inputName
    # PHASE FILE 
    filepick="{0}.pha".format(root)
    checkRequiredFiles(filepick)
    # VELOCITY FILE
    filevel="{0}.prm".format(root)
    checkRequiredFiles(filevel)
    # STATION FILE
    filesta="{0}.sta".format(root)
    checkRequiredFiles(filesta)
    # PARAMETER FILE
    filepar="default.cfg"
    checkRequiredFiles(filepar)
    # REFERENCE TIME
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
    # Run Hypoellipse in Windows
    if platform.system() == "Windows":
        with open("{0}_runHE.bat".format(root), "w") as f:
            f.write("utils\hypoellipseMain.exe < {0} > NUL\n ".format(filecom))
        cmd = "{0}_runHE.bat > NUL".format(root)
    # Run Hypoellipse in Linux
    elif platform.system() == "Linux":
        with open("{0}_runHE.sh".format(root), "w") as f:
            f.write("utils/hypoellipseMain < {0} 2>&1 >/dev/null\n ".format(filecom))
        cmd = "bash {0}_runHE.sh".format(root)
    # Run and Get statistics
    os.system(cmd)
    meanRMS, meanERH, meanERZ = getStatistic(root)
    # Remove unused files
    for f in ["{0}.1st".format(root),
            "{0}.2st".format(root),
            "{0}.3sc".format(root),
            "{0}.4sc".format(root),
            "{0}.5sc".format(root),
            "{0}.arc".format(root),
            "{0}.log".format(root),
            "{0}.sum".format(root),
            "{0}_runHE.bat".format(root),
            "{0}_runHE.sh".format(root),
            filecom]:
        if os.path.exists(f):
            os.remove(f)
    return meanRMS, meanERH, meanERZ        
