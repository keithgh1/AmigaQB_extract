# AmigaQB_extract
Amiga Quarterback ADF file extractor

This python script is designed to accept the Central Coast Software's Quarterback backup disks as .ADF files, identify the file markers within the file, and perform the LZW decompression algorithm on raw datastream, and write those decompressed files back into a qb_dump subdirectory of the current directory. 

The current file markers supported are CFM for compressed files, and FMRK for uncompressed files.

**You must create a qb_dump folder!**

## Dependencies

This script requires Python 3.6 or greater. There's nothing special or unique or exotic about the installation, and this script should function fine on Windows, OSX, and Linux. I developed it mostly using Jupyter Notebook, but tested it on Ubuntu 22.04. Also tested on Windows 11.

You'll want to install PIP, if you've somehow used Python without it. And then install [pathvalidate](https://pypi.org/project/pathvalidate/), [numpy](https://numpy.org/install/) and [pandas](https://pandas.pydata.org/docs/getting_started/index.html#getting-started) using PIP.

## More details
This backup software was a popular HDD backup solution for the Commodore Amiga. The genesis of this project was that I had a couple corrupted disks within my backup set, and Quarterback, despite having SOME builtin protection against it, fails as soon as it encounters some corrupted data. QB also requires a catalog which is stored on the first and last floppy in the backup set, in order to extract the files. My tool works on standalone disks within the set with or without the catalog.

There is support for a backup set spanning multiple disks. Once ADF'd, combine the ADF's into a single file, concatenating one disk after the others. The disk must be 901120 bytes long, and they should be in order. Once the ADF's are named properly, using cat within linux and redirecting the output is sufficient to build a massive disk file. Then, just execute this script against the file.

Example: Let's say your files are numbered like this 01.adf, 02.adf, and 03.adf. Your command to concatenate those together might look like this:
`cat *.adf > combined.adf`

Because of the way LZW builds a new dictionary for every file, any corruption within a current file makes the rest of the file unreadable. The data prior to the corruption is safe.

I have not done much testing on it especially regarding the numerous different versions of quarterback, but more is planned. It's possible that some minor modifications are needed to support it, please email me the details and a sample ADF if possible, and I'll try my best to fix it.

Source code to the original Quarterback tools are available here: https://gitlab.com/amigasourcecodepreservation/quarterback

## Usage

Please note that the backup file provided is typically a 901,120 byte .ADF file, or similar multiple-sized combined set of ADFs. This software does not yet work with a single large backup file that wasn't originally created as a series of floppy disks. This support should be easy enough to add, but the format is just different enough to require separate processing.

If you get really screwy catalog processing results with the default header length of 20, then switch it to 16 to see if it makes a difference.

```
usage: amigaqb_extract.py [-h] [--catalog {primary,backup,ignore}] [--version] [--header-length {16,20}] backup_file

Restore files from an Amiga Quarterback backup file.

positional arguments:
  backup_file           The backup filename to process

options:
  -h, --help            show this help message and exit
  --catalog {primary,backup,ignore}
                        Choose how to handle the catalog: 'primary' (default) will process the primary catalog at the beginning of the backup file,
                        'backup' will process the backup catalog, 'ignore' will ignore the catalog and dump all restored files into the root qb_dump
                        directory.
  --version             show program's version number and exit
  --header-length {16,20}
                        Specify the header length of DirFib entries: '16' bytes for entries with one file size, '20' bytes (default) for entries with
                        two file sizes.
```
## LZW Details

QB tools uses fairly standard LZW compression on its files. It uses a code-size of 9 bits minimum, 16 bits maximum. The original C and ASM source code for quarterback is available publically, and I'll link it in the future. Because of the 16-bit maximum code-size, there's a drawback to compression performance on files over a couple hundred kilobytes. If there are repeated patterns not yet learned by the time all ~65k codes are filled, then the LZW dictionary will get no bigger. I have plans to have the script report the compression efficiency for each file as it's extracted.

```
#define FIRST_CODE	258				/* First free entry */
#define CLEAR_CODE	257				/* Table clear output code */
#define EOF_CODE 	256				/* Last entry of file */
```

## Release History

Sept 10th, 2024: Version 0.4.2

* Minor update to address issue #8.
* Any directory of filename parsed from the backup file is now run through pathvalidate's sanitize_path()
* This prevents illegal characters from even being attempted to be written
* Also added was try/except blocks around all file operations. This prevents the script from failing and allows for more graceful handling

Sept 9th, 2024: Version 0.4.1

* Minor update to sanitize directory names and catch file system errors when creating directories. More work needs done here.
* Removed the writing of the debug file decryptedcat.bin

Sept 9th, 2024: Version 0.4.0

This version would have been delayed without the help from the Potato King from Reddit who figured out that the file catalog is encrypted even when encryption is disabled. Just realizing that fact, with a nudge to look in Monitor.C was exactly what I needed to add this support!

This release almost triples the lines of code of the original version! This is a massive rewrite which adds a ton of features:

* Adds primary backup catalog parsing, which now will create the original directories, and match file entries with markers
* Adds command line arguments for catalog support, including the ability to ignore a corrupted catalog, now using argparse
* Adds two different file catalog entry header types: 16-bytes and 20-bytes. If one doesn't work, just try the other.
* Rewrites the bit manipulation code extraction for compression, which broke under newer versions of python
* Checks the python version to insure required 'f' support
* Refactors a bunch of functions to make them easier to read, including added docstrings.
* Used tools like autopep8, vermin, and pylint to improve code quality

August 2024: Version 0.3.0

Minor changes including support for FMRK uncompressed files

Copyright (c) 2024 Keith Monahan
Licensed under the MIT License. See LICENSE file in the project root for full license information.
