# AmigaQB_extract
Amiga Quarterback backup extractor and data-recovery tool

This Python tool reads Central Coast Software's Quarterback hard-disk backups as `.ADF` floppy images, finds the files stored inside, decompresses them, and writes them back out with their original directory structure. It is built for **recovery**: it does its best to get your data out even when disks are damaged, the catalog is corrupt, or only part of the backup set survives.

All recovered files — plus a `_recovery_report.txt` manifest of what was recovered — are written to a `qb_dump` folder in the current directory, which is created automatically.

## Why
Quarterback was a popular HDD backup solution for the Commodore Amiga. The genesis of this project was that I had a couple of corrupted disks within my backup set, and Quarterback — despite some built-in protection — gives up as soon as it hits corrupted data. It also needs a catalog (stored on the first and last floppy of a set) in order to restore anything. This tool works on standalone disks within a set, with or without the catalog, and keeps going through damage rather than stopping at it.

## Supported formats
The format, Quarterback version, compression, encryption, and catalog layout are all detected automatically — in the common case there is nothing to specify.

- **Modern Quarterback v5.x / v6.x** ("Qb" disks): encrypted catalog; file markers `FMRK` (uncompressed) and `CFM` (LZW-compressed at any code size, 9–16 bits); disk-image markers `IMRK`/`CIM`; single- or multi-disk sets.
- **Antique Quarterback V4.x** ("QB" disks, ~1991): plaintext catalog, no per-file markers (file data is concatenated in catalog order).

## Dependencies
Python 3.6 or greater. Tested on Windows 11 and Ubuntu 22.04; it should be fine on macOS as well. Install the required packages (numpy, pathvalidate) with pip:

```
pip install -r requirements.txt
```

## Multiple disks
A Quarterback backup usually spans several floppies. Just pass all of the disk images (or a directory containing them) and the tool combines them in the correct order — no need to `cat` them together by hand:

```
python amigaqb_extract.py diskA.adf diskB.adf diskC.adf
python amigaqb_extract.py *.adf
python amigaqb_extract.py my_backup_set/
```

Disks are ordered by the disk-sequence number stored in each disk's header (not by filename or the order you list them), so a shell glob like `*.adf` that expands to `1.adf 10.adf 2.adf` still works. Each raw disk image must be 901120 bytes. The tool warns if a disk number is missing (incomplete set) or if the disks carry different backup date/time stamps (i.e. they look like they came from different backups). Files whose data spans a disk boundary are reassembled automatically.

You can still pass a single, already-concatenated file if you prefer:

```
cat *.adf > combined.adf  &&  python amigaqb_extract.py combined.adf
```

## Recovery features
The whole point of this tool is getting data off imperfect disks. Among other things, it will:

- recover files even when **no catalog at all** (neither primary nor backup) is present or readable — every file is saved from its data marker, with duplicate names kept unique so nothing is silently overwritten;
- fall back to the **backup catalog** on the last disk when the primary (first-disk) catalog is missing or corrupt;
- **brute-force the catalog encryption seed** if the byte that holds it is unreadable;
- recover a file marker whose tag byte is corrupted, and keep a single corrupt catalog entry from derailing the rest of the parse;
- on the antique format (which has no markers), repair a corrupt file size from the backup catalog, or re-anchor past it to the next recognizable file when there is no backup catalog;
- flag files that overlap suspected unreadable/filled sectors, and files Quarterback itself couldn't fully read at backup time.

Everything it did — including any files it could only recover partially — is recorded per-file in `qb_dump/_recovery_report.txt`.

Note on compressed files: because LZW builds a fresh dictionary per file, corruption inside a compressed file makes the rest of *that* file unreadable, but the data before the corruption is kept and every other file is unaffected.

## Usage
A backup input is typically a 901120-byte `.ADF` floppy image (or several of them). This tool does not yet handle a single large backup image that was not originally written as a series of floppies.

The catalog entry layout (one file size vs. two) is detected automatically, so you normally don't need `--header-length` at all. If automatic detection ever guesses wrong, you can still force `16` or `20` explicitly.

```
usage: amigaqb_extract.py [-h] [--catalog {primary,backup,ignore}] [--version] [--header-length {auto,16,20}] backup_file [backup_file ...]

Restore files from an Amiga Quarterback backup file.

positional arguments:
  backup_file           One or more backup inputs to process. Pass several disk .adf files, or a directory containing them, and they are
                        combined automatically in disk order - no need to 'cat' them together first. A single already-combined file also works.

options:
  -h, --help            show this help message and exit
  --catalog {primary,backup,ignore}
                        How to use the catalog: 'primary' (default) uses the catalog on the first disk; 'backup' uses the alternate catalog on the
                        last disk; 'ignore' uses no catalog at all and recovers every file from its data marker by filename (duplicate names are kept
                        unique so nothing is overwritten). 'primary' and 'backup' automatically fall back to this marker-only recovery if the catalog
                        is missing or unreadable - a catalog is never required to get the data out.
  --version             show program's version number and exit
  --header-length {auto,16,20}
                        Catalog entry layout: '16' for one file size, '20' for two file sizes. Default 'auto' detects it automatically by trying both
                        and picking the one that parses cleanly; only override if detection guesses wrong.
```

If you have a Quarterback version or configuration this doesn't handle, please email me the details and a sample ADF if you can, and I'll do my best to add support.

Source code to the original Quarterback tools is available here: https://gitlab.com/amigasourcecodepreservation/quarterback

## LZW Details

QB tools uses fairly standard LZW compression on its files, with a code size of 9 bits minimum and 16 bits maximum (the extractor reads each file's actual maximum from its marker). Because of the 16-bit maximum code size, there's a drawback to compression performance on files over a couple hundred kilobytes: if there are repeated patterns not yet learned by the time all ~65k codes are filled, the LZW dictionary will get no bigger.

```
#define FIRST_CODE	258				/* First free entry */
#define CLEAR_CODE	257				/* Table clear output code */
#define EOF_CODE 	256				/* Last entry of file */
```

## Release History

June 2026: Versions 0.5.0 – 0.17.0 — major recovery overhaul

A single concentrated effort took the tool from 0.4.4 to 0.17.0, turning a single-configuration extractor into a broad, corruption-tolerant recovery tool. The on-disk format was reverse-engineered against the original (now GPL) Quarterback source, and every change below was validated against a range of real and deliberately-corrupted test backups.

**More formats — all detected automatically (nothing to specify in the common case):**

* Antique Quarterback V4.x ("QB", ~1991): a completely different, marker-less format (plaintext catalog, file data concatenated in catalog order) — fully supported via its own extraction path.
* Disk-image backups: `IMRK` (uncompressed) and `CIM` (compressed) markers.
* Any LZW code size 9–16 bits, read per-file from the marker (was hard-coded to 16-bit only, which silently skipped 12–15-bit backups).
* Both catalog entry layouts (16-byte single-size and 20-byte two-size), auto-detected — `--header-length` is no longer needed.
* Backup format, Quarterback version, compression, encryption, and the encryption seed are all detected automatically.

**Multi-disk, done properly:**

* Pass several disk images, a shell glob, or a directory — they are combined for you, in disk order, by the sequence number in each disk's header. No more manual `cat`.
* Files whose data spans a disk boundary are reassembled byte-exactly.
* Warns on incomplete sets and on disks that don't appear to belong to the same backup.

**Corruption resilience — you should never need a perfect disk, or even a catalog, to get data out:**

* Marker-only recovery when there is no usable catalog at all — every file is saved from its data marker, with duplicate names kept unique so nothing is overwritten.
* Backup (`Qbc2`) catalog fallback when the first-disk catalog is missing or corrupt.
* Brute-forces the catalog encryption seed when the byte that holds it is on a bad sector.
* Recovers a file marker whose tag byte is corrupted, and re-syncs the catalog mid-stream so one bad entry doesn't lose everything after it.
* Hard/soft link entries are parsed correctly (the old parser ignored their link-target string, which silently corrupted any catalog containing a link).
* Antique format: a corrupt file size is repaired from the backup catalog, or re-anchored past to the next recognizable Amiga file when there is no backup catalog.
* Compressed streams return their valid leading data instead of crashing on corruption.
* Flags files Quarterback itself couldn't fully read at backup time, and files overlapping suspected unreadable/filled sectors.
* The catalog-to-data matcher was rewritten from a fragile fixed-position window to whole-sequence alignment, dramatically improving partial and last-disk-only recoveries.

**Output and usability:**

* A single consolidated analysis report at the start of every run (format, version, disks, compression, encryption, catalog, marker tally, warnings).
* A per-file `_recovery_report.txt` manifest recording the status and outcome of every recovered file.
* The `qb_dump` output folder is created automatically.
* Path components are sanitized individually, with placeholders for unnamed/garbage entries so the directory structure is preserved.

**Housekeeping:** dropped the unused `pandas` dependency and removed dead code; dependencies are now listed in `requirements.txt`.

Sept 22nd, 2024: Version 0.4.4

* Address issue #5 by applying the timestamps, if available in the associated dirfib entry, to directories and files.

Sept 22nd, 2024: Version 0.4.3

    Cross-Platform Path Handling Improvements:
        Eliminated hard-coded OS-type settings for path separators.
        Switched to using pathlib for all path manipulations, providing a more modern and Pythonic approach.
        Replaced custom generate_path function and os.path.join() with pathlib throughout the script.
        Ensured consistent and cleaner platform handling across different operating systems.

    Bug Fixes:
        Fixed an issue where certain print statements using f-strings could cause errors on older versions of Python.
        Ensured all f-strings are correctly formatted for compatibility with a wider range of Python versions.

This update enhances the code's robustness and compatibility across platforms while providing more reliable output handling. I tested this under Windows 11 and Ubuntu 22.04.

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
