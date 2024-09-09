# AmigaQB_extract
Amiga Quarterback ADF file extractor

This python script is designed to accept the Central Coast Software's Quarterback backup disks as .ADF files, identify the file markers within the file, and perform the LZW decompression algorithm on raw datastream, and write those decompressed files back into a qb_dump subdirectory of the current directory. 

The current file markers supported are CFM for compressed files, and FMRK for uncompressed files.

**You must create a qb_dump folder!**

## Dependencies

This script requires Python 3.6 or greater. There's nothing special or unique or exotic about the installation, and this script should function fine on Windows, OSX, and Linux. I developed it mostly using Jupyter Notebook, but tested it on Ubuntu 22.04. Also tested on Windows 11.

You'll want to install PIP, if you've somehow used Python without it. And then install [numpy](https://numpy.org/install/) and [pandas](https://pandas.pydata.org/docs/getting_started/index.html#getting-started) using PIP.

## More details
This backup software was a popular HDD backup solution for the Commodore Amiga. The genesis of this project was that I had a couple corrupted disks within my backup set, and Quarterback, despite having SOME builtin protection against it, fails as soon as it encounters some corrupted data. QB also requires a catalog which is stored on the first and last floppy in the backup set, in order to extract the files. My tool works on standalone disks within the set without the catalog.

There is support for a backup set spanning multiple disks. Once ADF'd, combine the ADF's into a single file, concatenating one disk after the others. The disk must be 901120 bytes long, and they should be in order. Once the ADF's are named properly, using cat within linux and redirecting the output is sufficient to build a massive disk file. Then, just execute this script against the file.

Example: Let's say your files are numbered like this 01.adf, 02.adf, and 03.adf. Your command to concatenate those together might look like this:
`cat *.adf > combined.adf`

Because of the way LZW builds a new dictionary for every file, any corruption within a current file makes the rest of the file unreadable. The data prior to the corruption is safe.

I have not done much testing on it especially regarding the numerous different versions of quarterback, but more is planned. It's possible that some minor modifications are needed to support it, please email me the details and a sample ADF if possible, and I'll try my best to fix it.

## LZW Details

QB tools uses fairly standard LZW compression on its files. It uses a code-size of 9 bits minimum, 16 bits maximum. The original C and ASM source code for quarterback is available publically, and I'll link it in the future. Because of the 16-bit maximum code-size, there's a drawback to compression performance on files over a couple hundred kilobytes. If there are repeated patterns not yet learned by the time all ~65k codes are filled, then the LZW dictionary will get no bigger. I have plans to have the script report the compression efficiency for each file as it's extracted.

```
#define FIRST_CODE	258				/* First free entry */
#define CLEAR_CODE	257				/* Table clear output code */
#define EOF_CODE 	256				/* Last entry of file */
```

## Release History

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
