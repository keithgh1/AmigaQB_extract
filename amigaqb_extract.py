"""
This script takes a Central Coast Software Quarterback backup disk, or a concatenated set of backup
disks, as input, and identifies which files are stored, and decompresses them into a qb_dump folder

Contact author Keith Monahan keith@techtravels.org with constructive feedback.
Bug reports should be filed as github issues. Please include the problem ADF backup file.
https://github.com/keithgh1/AmigaQB_extract
"""
import sys
import logging
# logging.basicConfig(FILENAME='qb_event.log', encoding='utf-8', level=logging.DEBUG)

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def hexdisplay(data):
    """
    Handles conversion of data to a hex string like this 0E FF C4...
    """
    output_data_string = ""

    for mydata in data:
        if len(hex(ord(mydata))) < 4:
            output_data_string += "0"

        output_data_string += hex(ord(mydata))[2:]
        output_data_string += " "

    return output_data_string

def get_code(buf, pos, code_size):
    """
    This func takes pos, which is a bit offset position from 0, returns code_size'd values from buf.
    """
    if code_size == 9:
        mask = 0x1ff
    elif code_size == 10:
        mask = 0x3ff
    elif code_size == 11:
        mask = 0x7ff
    elif code_size == 12:
        mask = 0xfff
    elif code_size == 13:
        mask = 0x1fff
    elif code_size == 14:
        mask = 0x3fff
    elif code_size == 15:
        mask = 0x7fff
    elif code_size == 16:
        mask = 0xffff
    else:

        print("Warning: Either code_size not set or not 9-16!!")
        return 0xDEADBEEF

    bytepos = int(pos / 8)
    bitpos = pos % 8

    # necessary for last byte processing. bytepos+2 exceeds end of buffer and throws index error

    if len(buf) > bytepos+2:
        val = ((buf[bytepos] >> bitpos) | (buf[bytepos+1] <<
               (8 - bitpos)) | (buf[bytepos+2] << (16 - bitpos))) & mask
    else:
        val = ((buf[bytepos] >> bitpos) | (
            buf[bytepos+1] << (8 - bitpos))) & mask

    logging.debug("pos: %s codesize: %s val: %s",pos,code_size,val)
    return val


def uncompressme(buf):
    """
    This function performs the lzw decompression and makes up most of the
    complexity of the overall script. It takes a raw LZW datastream in
    as byte buffer, converts to a stream of bits, and then extracts
    variable-sized codes, and decompresses them.

    See https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
    https://rosettacode.org/wiki/LZW_compression
    """
    bits = np.unpackbits(buf)

    next_code = 258
    decompressed_data = ""
    mystring = ""

    code_size = 9

    # 2^9 - 1
    maximum_table_size = 511

    # Building and initializing the dictionary
    # ASCII 0-255 in positions 0-255

    dictionary_size = 256
    dictionary = dict([(x, chr(x)) for x in range(dictionary_size)])

    # LZW Decompression algorithm

    i = 0

    while (i < (len(bits)-9)):

        # Don't allow code_size grow larger than 16-bits.
        if next_code > maximum_table_size and code_size != 16:

            # need to bump the bitpos by 9, essentially skipping a 9-bit word
            # Also bump code size by 1, and recalculate the maximum code
            # Only skip the 9-10 transition? not others? OK!
            if code_size == 9:
                i += code_size

            code_size += 1
            maximum_table_size = pow(2, int(code_size)) - 1

            logging.debug("Code size change at bits: %s",i)
            logging.debug("code size is %s",code_size)
            logging.debug("Approximate decompress size is %s"
                          ,len(decompressed_data))

        code = get_code(buf, i, code_size)

        if code == 0xDEADBEEF:
            return str.encode("DEADBEEF")

        if code > next_code:
            logging.debug("Code read is higher than it should be. code is %s next_code is %s"
                          ,code, next_code)

        if code not in dictionary:
            dictionary[code] = mystring + (mystring[0])

        # This generates a ton of log entries
        # logging.debug("Decomp-pos:"+str(len(decompressed_data)))
        # logging.debug("out:"+hexdisplay(dictionary[code]))

        decompressed_data += dictionary[code]

        if len(mystring) != 0:
            dictionary[next_code] = mystring + (dictionary[code][0])
            next_code += 1
            logging.debug("Nextcode is now %s",next_code)

        mystring = dictionary[code]

        i += code_size

    output_data_string = ""

    # Shouldn't we be using hexdisplay() for this?
    for data in decompressed_data:
        if len(hex(ord(data))) < 4:
            output_data_string += "0"

        output_data_string += hex(ord(data))[2:]
        output_data_string += " "

    return bytearray.fromhex(output_data_string)

d = {'Offset': [], 'FILENAME': [], 'Filesize1': [],
     "Filesize2": [], "Filecontent": []}
df = pd.DataFrame(data=d)

# yes, yes, I should be use argparse
if len(sys.argv) != 2:
    print("Usage: python3 amigaqb_extract.py backup-filename-to-process")
    sys.exit()

fullfile = np.fromfile(sys.argv[1], dtype=np.ubyte)
# fullfile = np.fromfile("testfile1.adf", dtype=np.ubyte)

logging.debug("Opened file: %s File size is %s",sys.argv[1],len(fullfile))

# need to process the multidisk concatenations before file offsets because file offsets are going
# to change as a result of the disk header deletions

Qb_list = []

if len(fullfile) < 901121:
    print("Single disk detected.")
else:
    print("Multidisk detected, looking for QB ids...")

    searchval = [0x51, 0x62]
    N = len(searchval)
    possibles = np.where(fullfile == searchval[0])[0]

    for p in possibles:
        check = fullfile[p:p+N]
        if np.all(check == searchval):
            # p has to be on an amiga disk boundary number of bytes
            # file header is 16 bytes, so let's add that for removal
            if not p % 901120:

                Qb_list.append((p, p+16))

    print("QB id's found:", len(Qb_list))
    logging.debug("QB ids found: %s",len(Qb_list))
    logging.debug(Qb_list)
    print(Qb_list)

    if len(Qb_list) > 0:
        fullfile = np.delete(fullfile, np.concatenate(
            [np.arange(start, end) for start, end in Qb_list]))

# look for CFM 0x90 pattern in file headers
# What would happen if that pattern shows up in the regular compressed data?
# What's the probability of it? Pretty low me thinks?

searchval = [0x43, 0x46, 0x4D, 0x90]
N = len(searchval)
possibles = np.where(fullfile == searchval[0])[0]

offset_list = []

for p in possibles:
    check = fullfile[p:p+N]
    if np.all(check == searchval):
        offset_list.append(p)

logging.debug(offset_list)
filelist = []

print("Found the following files:\n")
for offset in offset_list:

    FILENAME = "".join([chr(item) for item in fullfile[offset+4:offset+34]])
    # print(FILENAME)

    filesize1bits = np.unpackbits(fullfile[offset+36:offset+40])
    filesize1 = filesize1bits.dot(2**np.arange(filesize1bits.size)[::-1])

    filelist.append([offset, FILENAME.rstrip('\x00'),
                    filesize1, 0, 'Nocontent', '', ''])

df = pd.DataFrame(filelist, columns=[
                  'Offset', 'FILENAME', 'Filesize1', 'Filesize2', 'State', 'CompressedData',
                  'UncompressedData'])
print(df[['Offset', 'FILENAME', 'Filesize1']])

for j in range(0, len(filelist)):
    CFM_offset = filelist[j][0]

    END_LZW_DATASTREAM = 0

    if j == len(filelist)-1:
        END_LZW_DATASTREAM = len(fullfile)
    elif j < len(filelist):
        NEXT_CFM_offset = filelist[j+1][0]

    START_LZW_DATASTREAM = CFM_offset+40

    # ie the end of the datastream doesn't end at the end of current file
    if END_LZW_DATASTREAM == 0:

        # I hate this section of code but heck if I can figure out how to find the filesize1
        # identified after the FILENAME and look for it to determine end of the raw LZW
        # datastream. This should work for all cases
        # This is necessary because there's a variable amount of padding
        # (none, 1, 2, 3 bytes) between filesize2 and the next CFM block. This changes
        # the end of data stream location depending on padding length

        candidate1bits = np.unpackbits(
            fullfile[NEXT_CFM_offset-4:NEXT_CFM_offset])
        candidate1size = candidate1bits.dot(
            2**np.arange(candidate1bits.size)[::-1])

        candidate2bits = np.unpackbits(
            fullfile[NEXT_CFM_offset-5:NEXT_CFM_offset-1])
        candidate2size = candidate2bits.dot(
            2**np.arange(candidate2bits.size)[::-1])

        candidate3bits = np.unpackbits(
            fullfile[NEXT_CFM_offset-6:NEXT_CFM_offset-2])
        candidate3size = candidate3bits.dot(
            2**np.arange(candidate3bits.size)[::-1])

        candidate4bits = np.unpackbits(
            fullfile[NEXT_CFM_offset-7:NEXT_CFM_offset-3])
        candidate4size = candidate4bits.dot(
            2**np.arange(candidate4bits.size)[::-1])

        if candidate1size == filelist[j][2]:
            END_LZW_DATASTREAM = NEXT_CFM_offset-4

        elif candidate2size == filelist[j][2]:
            END_LZW_DATASTREAM = NEXT_CFM_offset-5

        elif candidate3size == filelist[j][2]:
            END_LZW_DATASTREAM = NEXT_CFM_offset-6

        elif candidate4size == filelist[j][2]:
            END_LZW_DATASTREAM = NEXT_CFM_offset-7

        else:
            # This shouldn't happen....
            END_LZW_DATASTREAM = 99999999

    # sys.stdout.write("Processing ")
    print(filelist[j][1], " at Offset ",
          START_LZW_DATASTREAM, "-", END_LZW_DATASTREAM)

    logging.debug("%s at Offset %s - %s",filelist[j][1],
                  START_LZW_DATASTREAM, END_LZW_DATASTREAM)

    # raw datastream continues at qb02 position + 16

    filelist[j][6] = uncompressme(
        fullfile[START_LZW_DATASTREAM:END_LZW_DATASTREAM])

    if len(filelist[j][6]) == filelist[j][2]:
        print("Size check good.\n")
        filelist[j][5] = "Decompressed OK"
    else:
        # print("LZW decompression returned the wrong amount of data\n")

        # For like 5 or 6% of my test data, we need to bump the end position by 1
        # I have no clue why this is, maybe something to do more with variable padding?
        # This fixes those special cases

        filelist[j][6] = uncompressme(
            fullfile[START_LZW_DATASTREAM:END_LZW_DATASTREAM+1])
        if len(filelist[j][6]) == filelist[j][2]:
            print("Size check good.\n")
            filelist[j][5] = "Decompressed OK on +1 retry"
        else:
            print("ERROR Size expected:", filelist[j][2])
            print("ERROR Size returned:", len(filelist[j][6]))
            if filelist[j][6] == "DEADBEEF":
                print("Bad code_size detected")
            filelist[j][5] = "Bad Size"

    with open("qb_dump/"+filelist[j][1], "wb") as binary_file:
        binary_file.write(filelist[j][6])
