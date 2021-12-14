from LatLon import lat_lon as ll
import os, platform

"""
Script for running hypo71 program.

inputs:
- hypoellipse phase file,
- hypoellipse station file,
- hypoellipse velocity model file,

LogChange:
13-Dec-2021 > init.

"""

# Get mean of a list
def mean(a):
    return sum(a)/len(a)

# Add default values (RESET TEST) to hypo71 input file
def addHypo71Defaults(fileObject):
    fileObject.write("HEAD                     GENERATED USING 'hypo71.py' CODE\n")
    fileObject.write("RESET TEST(01)=0.1\n")
    fileObject.write("RESET TEST(02)=10.\n")
    fileObject.write("RESET TEST(03)=1.5\n")
    fileObject.write("RESET TEST(04)=0.05\n")
    fileObject.write("RESET TEST(05)=5.\n")
    fileObject.write("RESET TEST(06)=4.\n")
    fileObject.write("RESET TEST(11)=40.\n")
    fileObject.write("\n")

# Add hypo71 station information to hypo71 input file
def addHypo71Stations(fileObject, stationDict):
    for station in stationDict:
        fileObject.write(stationDict[station])
    fileObject.write("\n")

# Add hypo71 velocity model to hypo71 input file
def addHypo71VelocityModel(fileObject, velocityModel):
    for layer in velocityModel:
        fileObject.write(layer)
    fileObject.write("\n")

# Add hypo71 control line to hypo71 input file
def addHypo71ControlLine(fileObject, controlLine):
    fileObject.write(controlLine)

# Add hypo71 phase information to hypo71 input file
def addHypo71Phase(objectFile, phaseFile):
    with open(phaseFile) as f:
        for l in f:
            objectFile.write(l)

# Parse hypoellipse station file
def parseHypoellipseStation(hypoellipseStationFile):
    stationDict = {}
    with open(hypoellipseStationFile) as f:
        for l in f:
            if "*" not in l:
                staCode = l[:4].strip()
                latDeg, latMin = int(l[4:6]), float(l[7:12])
                lat = ll.Latitude(latDeg, latMin)
                lonDeg, lonMin = int(l[13:16]), float(l[17:22])
                lon = ll.Longitude(lonDeg, lonMin)
                elv = int(l[22:27])
                stationDict[staCode] = "  {0:4s}{1:2d}{2:05.2f}N {3:2d}{4:05.2f}E{5:4d}\n".format(
                    staCode, int(lat.decimal_degree), lat.decimal_minute, int(lon.decimal_degree), lon.decimal_minute, elv)
    return stationDict

# Parse hypoellipse velocity model file
def parseHypoellipseVelocity(hypoellipseVelocityModelFile):
    velocityModel = []
    with open(hypoellipseVelocityModelFile) as f:
        for l in f:
            Vp = float(l.split()[1])
            Z = float(l.split()[2])
            velocityModel.append("  {0:5.3f} {1:6.3f}\n".format(Vp, Z))
    return velocityModel

# Parse control file file
def parseHypo71ControlLine(hypoellipseVelocityModelFile):
    with open(hypoellipseVelocityModelFile) as f:
        for l in f:
            VpVs = float(l.split()[3])
            break
    trialDepth = 10
    xNear = 75
    xFar = 400
    controlLine = "{0:4d}.{1:4d}.{2:4d}.{3:5.2f}    4    0    0    1    1    0    0 0111\n".format(trialDepth, xNear, xFar, VpVs)
    return controlLine

# Get statistics of hypo71 outputs
def getStatistic(rootName):
    RMS, ERH, ERZ = [], [], []
    with open("{0}_h71.out".format(rootName)) as f:
        next(f)
        for l in f:
            if "*" not in l:
                try:
                    rms = float(l[62:67])
                    erh = float(l[67:72])
                    erz = float(l[72:77])
                    RMS.append(rms)
                    ERH.append(erh)
                    ERZ.append(erz)
                except ValueError:
                    continue
    return mean(RMS), mean(ERH), mean(ERZ)

# Create hypo71 phase file
def createHypo71PhaseFile(rootName):
    hypo71PhaseFile = "{0}_h71.pha".format(rootName)
    hypoellipseStationFile = "{0}.sta".format(rootName)
    hypoellipseVelocityModelFile = "{0}.prm".format(rootName)
    hypoellipsePhaseFile = "{0}.pha".format(rootName)
    with open(hypo71PhaseFile, "w") as f:
        addHypo71Defaults(f)
        stationDict = parseHypoellipseStation(hypoellipseStationFile)
        addHypo71Stations(f, stationDict)
        velocityModel = parseHypoellipseVelocity(hypoellipseVelocityModelFile)
        addHypo71VelocityModel(f, velocityModel)
        controlLine = parseHypo71ControlLine(hypoellipseVelocityModelFile)
        addHypo71ControlLine(f, controlLine)
        addHypo71Phase(f, hypoellipsePhaseFile)

# Create hypo71 input file
def createHypo71InputFile(rootName):
    outName = "{0}.inp".format(rootName)
    with open(outName, "w") as f:
        f.write("{0}_h71.pha\n".format(rootName))
        f.write("{0}_h71.prt\n".format(rootName))
        f.write("{0}_h71.out\n".format(rootName))
        f.write("{0}_h71.res\n".format(rootName))
        f.write("\n\n")

# Run hypo71 program
def runHypo71(rootName):
    if platform.system() == "Linux":
        cmd = "utils/hypo71Main < {0}.inp > /dev/null".format(rootName)
    elif platform.system() == "Windows":
        cmd = "utils/hypo71Main.exe < {0}.inp > NUL".format(rootName)
    createHypo71PhaseFile(rootName)
    createHypo71InputFile(rootName)
    os.system(cmd)
    RMS, ERH, ERZ = getStatistic(rootName)
    for i in ["{0}_h71.pha".format(rootName),
              "{0}_h71.out".format(rootName),
              "{0}_h71.prt".format(rootName),
              "{0}_h71.res".format(rootName),
              "{0}.inp".format(rootName)]:
        if os.path.exists(i):
            os.remove(i)
    return RMS,ERH,ERZ

# Run
# rootName = "hypo71"
# runHypo71(rootName)