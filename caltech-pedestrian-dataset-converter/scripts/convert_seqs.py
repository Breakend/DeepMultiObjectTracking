import os.path
import fnmatch
import shutil

# --------- Configurable parameters start ----------- #
# Does not write files on disk. Just prints their names on console.
# Used for generating a file containing the names of images relative to the
# base path(without extension, but includes prefix e.g. 'set01/V000/' )
print_names = 0

# Make interval 1 if want to print all file names
# Used for generating 1x, 10x training/test set for caltech
if print_names:
    interval = 30

# A list of all sets to be parsed
#sets = ['set00', 'set01', 'set02', 'set03', 'set04',
#'set05', 'set06', 'set07', 'set08','set09', 'set10']

sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05', 'set06', 'set07', 'set08','set09', 'set10'] #train
#sets = ['set06', 'set07', 'set08','set09', 'set10'] #test

#sourcedir must have directories set00, set01 etc
sourcedir = '/Users/Weili/Documents/McGill/Masters/COMP765/Project/caltech-pedestrian-dataset-converter/data'
#outdir will have directories set00/V000/*jpg, set00/V001/ .. set01/V000/ etc
outdir = '/Users/Weili/Documents/McGill/Masters/COMP765/Project/caltech-pedestrian-dataset-converter/data/images'
# --------- Configurable parameters end ----------- #

def open_save(seq_file, outdir, set, seq_name):
    # read .seq file
    with open(seq_file,'rb') as f:
        string = str(f.read())
    # split .seq file into segment with the image prefix
    splitstring = '\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46'
    strlist=string.split(splitstring)

    # create directories
    set_out_dir = os.path.join(outdir, set);
    seq_out_dir = os.path.join(outdir, set, seq_name.split('.')[0]);
    if not os.path.exists(set_out_dir): os.mkdir(set_out_dir)
    if not os.path.exists(seq_out_dir): os.mkdir(seq_out_dir)

    # deal with file segment, every segment is an image except the first one
    # Skip the first one, which is filled with .seq heade
    for idx, img in enumerate(strlist[1:]):
        filename = str(idx) + '.jpg'
        filenamewithpath=outdir+'/'+ set +'_'+ seq_name[:-4] +'_'+ filename
        if print_names:
            if (idx % interval) == 0:
                print os.path.join(set, seq_name, filename.split('.')[0])
        else:
            with open(filenamewithpath,'wb+') as i:
                i.write(splitstring + img)

if __name__=="__main__":
    if not os.path.exists(outdir): os.mkdir(outdir)

    for set in sorted(sets):
        print 'Parsing ', set,
        setdir = os.path.join(sourcedir, set);
        if not os.path.exists(setdir):
            print 'Cannot find directory: %s. Skipping.' % (setdir)
            continue

        for seq_name in sorted(os.listdir(setdir)):
            if seq_name.endswith(".seq"):
                open_save(os.path.join(setdir, seq_name), outdir, set, seq_name)
        print 'Done'
