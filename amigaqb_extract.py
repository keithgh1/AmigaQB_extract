"""
This script takes a Central Coast Software Quarterback backup disk, or a set of backup disks, as
input, identifies which files are stored, and decompresses them into a "qb_dump" folder in the
current directory (created automatically) along with a _recovery_report.txt manifest

Contact author Keith Monahan keith@techtravels.org with constructive feedback.
Bug reports should be filed as github issues. Please include the problem ADF backup file.
https://github.com/keithgh1/AmigaQB_extract

Copyright (c) 2024-2026 Keith Monahan
Licensed under the MIT License. See LICENSE file in the project root for full license information.
"""
import argparse
import sys
import os
import re
import logging
import struct
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from pathvalidate import sanitize_filepath, sanitize_filename


# logging.basicConfig(FILENAME='qb_event.log', encoding='utf-8', level=logging.DEBUG)
__version__ = "0.18.0"

# Minimum required version
REQUIRED_PYTHON = (3, 6)

DEFAULT_PATH = "qb_dump"

# Decryption table from the monitor.C code
decrypt_table = [
    151, 32, 127, 11, 234, 174, 21, 110, 67, 163, 203, 154, 13, 1, 171, 213,
    103, 56, 130, 18, 177, 134, 188, 146, 48, 88, 211, 167, 111, 227, 140, 243,
    120, 43, 250, 62, 76, 182, 253, 149, 193, 181, 135, 36, 27, 229, 143, 0,
    162, 220, 52, 85, 192, 196, 83, 25, 159, 246, 152, 6, 199, 138, 71, 208,
    16, 8, 125, 169, 148, 179, 93, 248, 108, 218, 186, 47, 29, 39, 145, 57,
    44, 230, 3, 96, 216, 119, 205, 175, 35, 65, 254, 172, 183, 54, 10, 197,
    128, 73, 31, 201, 42, 15, 46, 224, 244, 129, 180, 123, 156, 236, 158, 106,
    101, 212, 126, 12, 89, 202, 217, 69, 40, 20, 113, 33, 223, 232, 195, 235,
    118, 141, 70, 238, 84, 79, 23, 64, 209, 133, 24, 222, 55, 94, 105, 207,
    63, 66, 115, 241, 77, 61, 17, 92, 189, 198, 142, 75, 38, 98, 87, 170,
    97, 252, 147, 60, 245, 82, 53, 74, 184, 247, 251, 221, 90, 155, 176, 237,
    242, 51, 81, 100, 239, 59, 166, 225, 72, 153, 7, 41, 190, 116, 34, 2,
    187, 132, 144, 114, 117, 204, 30, 22, 86, 80, 139, 104, 9, 215, 178, 91,
    122, 233, 5, 231, 161, 28, 214, 49, 137, 78, 168, 102, 26, 112, 19, 185,
    150, 226, 164, 255, 45, 68, 206, 37, 173, 124, 4, 50, 219, 157, 240, 131,
    249, 191, 210, 136, 99, 95, 200, 228, 165, 109, 160, 194, 58, 121, 14, 107
]

# Vectorized form of the decryption table. The cipher is a stateless byte map,
# so a whole buffer decrypts as one numpy gather (see decrypt_data) instead of a
# per-byte Python loop.
DECRYPT_TABLE = np.array(decrypt_table, dtype=np.uint8)

# set flags for the DirFib structure, but used outside the class, so necessary here
FLAG_DIR_MASK = 128
FLAG_SEL_MASK = 64
FLAG_ERR_MASK = 32
FLAG_ODD_MASK = 16    # Obsolete, reserved
FLAG_PART_MASK = 8
FLAG_HLINK_MASK = 4
FLAG_SLINK_MASK = 2
FLAG_BITS_MASK = 1    # Backup-Compress or not, Restore-read-protected?

# Compression flag bits, from QB.h. The compressed-file marker is the ASCII
# "CFM" followed by one flag byte = COMP_ON_FLAG | maxBits.
MIN_COMPBIT = 12      # QB.h MIN_COMPBIT
MAX_COMPBIT = 16      # QB.h MAX_COMPBIT (this build's ceiling)
COMP_BIT_MASK = 0x3F  # low 6 bits = LZW max code size
COMP_ON_FLAG = 0x80   # set when the file is compressed
COMP_DEV_FLAG = 0x40  # device/image compression (reserved here)

# Backup format identification, from the first two bytes of disk 1.
QB_MODERN_ID = b'Qb'  # 0x5162 - v5.x/6.x: encrypted catalog, FMRK/CFM markers
QB_ANTIQUE_ID = b'QB'  # 0x5142 - V4.x: plaintext catalog, no markers

# Each disk after the first carries a 16-byte continuation header
# (QBOldFirstCylHdr 14 bytes + a 2-byte extraLen), per Restore.c SkipToDisk.
MODERN_CONT_HEADER = 16

# Antique (V4.x) on-disk constants, from quarterback/historical/qb_v1991/V4.0.
ANTIQUE_HEADER_SIZE = 14   # 'QBnn' + diskNum + altWrap + date(4) + time(4)
ANTIQUE_ENTRY_HEADER = 14  # size(4) date(2) time(2) ticks(2) filcnt(2) prot(1) flags(1)
ANTIQUE_FLAG_DIR = 0x80    # df_Flags bit 7 = directory
ANTIQUE_NAME_MAX = 30      # df_Name max length before the null
ANTIQUE_NOTE_MAX = 80      # df comment max length before the null


def log_phase(message):
    """
    Announce a processing phase on stderr. Kept off stdout so the analysis
    report (and any piped output) stays clean, but always visible so the user
    can see what the tool is doing at every step - nothing runs silently.
    """
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


class Progress:
    """
    Minimal, dependency-free progress indicator for the long loops
    (decompression, file writing), so the tool never just sits there silently.

    On an interactive terminal it rewrites a single line in place (throttled, so
    it never floods the console). When stderr is redirected or piped it prints
    nothing per item and only a final summary line - so it never spams a log or
    writes carriage returns into a captured stream. All output is ASCII and goes
    to stderr, leaving stdout for the report.
    """

    def __init__(self, label, total=None):
        self.label = label
        self.total = total
        self.n = 0
        self.tty = sys.stderr.isatty()
        self._last = 0.0

    def _bar(self):
        return f"{self.n}/{self.total}" if self.total else str(self.n)

    def update(self, n=1):
        self.n += n
        if self.tty:
            now = time.monotonic()
            if now - self._last >= 0.1:
                self._last = now
                sys.stderr.write(f"\r  {self.label}: {self._bar()}   ")
                sys.stderr.flush()

    def done(self):
        if self.tty:
            sys.stderr.write(f"\r  {self.label}: {self._bar()} - done\n")
        else:
            sys.stderr.write(f"  {self.label}: {self._bar()}\n")
        sys.stderr.flush()


def is_directory(flags):
    """A DirFib is a real directory only if the DIR bit is set and it is not a
    hard/soft link (a linked directory is an entry to recreate as a link, not a
    container to descend into). Mirrors the IS_DIR macro in the QB source."""
    return bool(flags & FLAG_DIR_MASK) and not (
        flags & (FLAG_HLINK_MASK | FLAG_SLINK_MASK))


def safe_component(name, fallback):
    """
    Sanitize a single path component (one file or directory name) with
    pathvalidate. If the name is empty or sanitizes away to nothing, return
    `fallback` instead, so an unnamed/garbage entry still gets a real folder or
    file and the directory structure beneath it is preserved rather than lost.
    """
    try:
        clean = sanitize_filename(name).strip()
    except (ValueError, TypeError):
        clean = ''
    if not clean or clean in ('.', '..'):
        return fallback
    return clean


class DirFib:
    """
    Represents a directory entry in the backup file.

    Attributes:
        df_size1 (int): The size of the file in bytes (part 1).
        df_size2 (int): The size of the file in bytes (part 2). Note this is unused in certain versions/configs
        df_days (int): The number of days since January 1, 1978.
        df_minutes (int): The number of minutes since midnight.
        df_ticks (int): The number of ticks (1/50th of a second) since the last minute.
        df_filcnt (int): The number of files in the directory.
        df_prot (int): The protection bits of the file.
        df_flags (int): The flags associated with the file.
        df_name (str): The name of the file.
        df_comment (str): The comment associated with the file.
        date (datetime): The actual date calculated from df_days.
        active_flags (list): The list of active flags based on df_flags.

    Methods:
        date_from_days_since_1978(days): Converts the number of days since 1978 to an actual date.
        from_bytes(byte_list, offset): Creates a DirFib instance from a list of bytes.
        _extract_null_terminated_string(byte_list, offset): Extracts a null-terminated string from a byte list.
        get_active_flags(flags): Returns a list of active flags based on the given flags value.
        flags_to_string(): Converts the active flags to a string representation.
    """

    def __init__(self, df_size1, df_size2, df_days, df_minutes,
                 df_ticks, df_filcnt, df_prot, df_flags, df_name, df_comment):
        self.df_size1 = df_size1
        self.df_size2 = df_size2
        self.df_days = df_days
        self.df_minutes = df_minutes
        self.df_ticks = df_ticks
        self.df_filcnt = df_filcnt
        self.df_prot = df_prot
        self.df_flags = df_flags
        self.df_name = df_name
        self.df_comment = df_comment
        # Set True when the entry is structurally sound but its datestamp is
        # out of range (kept by parse_dir_fibs rather than discarded); surfaced
        # as 'suspect-datestamp' in the recovery report.
        self.df_suspect_date = False

        # Convert df_days to an actual date
        self.date = self.date_from_days_since_1978(df_days)

        # Decode flags
        self.active_flags = self.get_active_flags(df_flags)

    @staticmethod
    def date_from_days_since_1978(days):
        """Converts the number of days since January 1, 1978 to a target date."""
        # January 1, 1978 as the starting date
        base_date = datetime(1978, 1, 1)

        # Calculate the target date by adding the number of days
        target_date = base_date + timedelta(days=days)

        return target_date

    @staticmethod
    def from_bytes(byte_list, offset, header_length='20'):
        """Converts a list of bytes to a DirFib instance with variable header length."""
        byte_stream = bytes(byte_list)

        # Determine the format based on the header length
        if header_length == '20':
            format_string = '>iiHHHHbB'  # 20 bytes, with two file sizes
        else:
            format_string = '>iHHHHbB'   # 16 bytes, with one file size

        fixed_size = struct.calcsize(format_string)
        fields = struct.unpack_from(format_string, byte_stream, offset)
        offset += fixed_size

        # Assign sizes based on the header length
        if header_length == '20':
            df_size1, df_size2, *remaining_fields = fields
        else:
            df_size1 = fields[0]
            df_size2 = 0  # Set df_size2 to zero if only one size is present
            remaining_fields = fields[1:]

        # remaining_fields = [days, minutes, ticks, filcnt, prot, flags]
        df_flags = remaining_fields[-1]

        # Extract null-terminated strings df_name and df_comment
        df_name = DirFib._extract_null_terminated_string(byte_list, offset)
        offset += len(df_name) + 1

        df_comment = DirFib._extract_null_terminated_string(byte_list, offset)
        offset += len(df_comment) + 1

        # Hard/soft link entries store a third null-terminated string after the
        # comment: the link target. It carries no file data, but it MUST be
        # consumed or every following catalog entry is misaligned. (The original
        # parser ignored it, which desynced catalogs containing links.)
        #
        # A corrupt entry can have the link bits set spuriously, though, and
        # consuming a bogus link target would itself desync everything after.
        # So consume it only if doing so leaves the *next* entry looking valid;
        # otherwise treat the link bits as corruption and don't.
        if df_flags & (FLAG_HLINK_MASK | FLAG_SLINK_MASK):
            link_name = DirFib._extract_null_terminated_string(byte_list, offset)
            after_link = offset + len(link_name) + 1
            if DirFib._next_entry_plausible(byte_list, after_link, header_length):
                offset = after_link

        # Create a DirFib instance
        return DirFib(df_size1, df_size2, *remaining_fields, df_name, df_comment), offset

    @staticmethod
    def _next_entry_plausible(byte_list, offset, header_length):
        """
        Peek at the fixed header that would begin at `offset` and report whether
        it looks like a real catalog entry (valid datestamp). Used to decide
        whether a link entry's target string should be consumed. The end of the
        buffer counts as plausible (nothing more to misalign).
        """
        fmt = '>iiHHHHbB' if header_length == '20' else '>iHHHHbB'
        size = struct.calcsize(fmt)
        if offset >= len(byte_list):
            return True
        if offset + size > len(byte_list):
            return False
        fields = struct.unpack_from(fmt, bytes(byte_list), offset)
        days, minutes, ticks = (fields[2], fields[3], fields[4]) \
            if header_length == '20' else (fields[1], fields[2], fields[3])
        return 0 <= days < 25000 and 0 <= minutes < 1440 and 0 <= ticks < 3000

    @staticmethod
    def _extract_null_terminated_string(byte_list, offset, max_len=1024):
        """
        Extract a null-terminated string starting at `offset`, retaining extended
        characters and sanitizing only unprintable ones.

        The scan is bounded to `max_len` bytes. A catalog name/comment/link
        target is short, so no terminator within that span means this isn't a
        real entry (garbage, or a wrong alignment) - raise so the caller resyncs
        or stops. Bounding the scan is what stops a constant-byte region (which
        contains no null at all) from turning every parse attempt into a
        full-buffer scan - the all-zeros / uniform-fill hang.
        """
        n = len(byte_list)
        limit = min(n, offset + max_len)
        # bytes/bytearray expose a C-level search; fall back to a loop otherwise.
        if hasattr(byte_list, 'find'):
            end = byte_list.find(b'\x00', offset, limit)
            if end == -1:
                end = limit
        else:
            end = offset
            while end < limit and byte_list[end] != 0x00:
                end += 1
        if end >= limit:
            raise ValueError("Null-terminated string not found")

        # Decode string using ISO-8859-1, replacing only invalid characters
        decoded_string = bytes(byte_list[offset:end]).decode(
            'iso-8859-1', errors='replace')

        # Keep printable characters, including extended characters from Latin-1
        sanitized_string = ''.join(
            c if c.isprintable() else '_' for c in decoded_string)

        return sanitized_string

    def get_active_flags(self, flags):
        """
        Returns a list of active flags based on the given flags value.
        """
        active_flags = []
        if flags & FLAG_DIR_MASK:
            active_flags.append('DIR')
        if flags & FLAG_SEL_MASK:
            active_flags.append('SEL')
        if flags & FLAG_ERR_MASK:
            active_flags.append('ERR')
        if flags & FLAG_ODD_MASK:
            active_flags.append('ODD')
        if flags & FLAG_PART_MASK:
            active_flags.append('PART')
        if flags & FLAG_HLINK_MASK:
            active_flags.append('HLINK')
        if flags & FLAG_SLINK_MASK:
            active_flags.append('SLINK')
        if flags & FLAG_BITS_MASK:
            active_flags.append('BITS')
        return active_flags

    def flags_to_string(self):
        """
        Converts the active flags to a string representation.

        Returns:
            str: A comma-separated string of active flags. If there are no active flags, returns 'None'.
        """
        return ', '.join(self.active_flags) if self.active_flags else 'None'

# Function to loop through the byte list and parse multiple DirFib structures


def _name_looks_valid(dir_fib):
    """The entry has a plausible name (non-empty, not over-long)."""
    return bool(dir_fib.df_name) and len(dir_fib.df_name) <= 30


def _datestamp_in_range(dir_fib):
    """The entry's Amiga datestamp fields are all in range."""
    return (0 <= dir_fib.df_days < 25000 and 0 <= dir_fib.df_minutes < 1440
            and 0 <= dir_fib.df_ticks < 3000)


def _entry_looks_valid(dir_fib):
    """A parsed entry is plausible if its datestamp is in range and it has a
    reasonable name. Used to know where a catalog ends when there is no marker
    boundary after it (the backup catalog) and as a general sanity check."""
    return _datestamp_in_range(dir_fib) and _name_looks_valid(dir_fib)


def _find_catalog_resync(byte_list, start, header_length, window=8192, need=3):
    """
    After a corrupt catalog entry has thrown the variable-length stream out of
    alignment, scan forward for the offset where parsing re-locks: the first
    position from which `need` consecutive entries all look valid. Requiring a
    run of valid entries avoids false re-locks. Returns the offset or None.
    """
    n = len(byte_list)
    for cand in range(start, min(n, start + window)):
        off, ok = cand, 0
        try:
            for _ in range(need):
                fib, off = DirFib.from_bytes(byte_list, off, header_length)
                if not _entry_looks_valid(fib):
                    break
                ok += 1
        except (ValueError, struct.error, IndexError):
            pass
        if ok >= need:
            return cand
    return None


# Hard bounds that keep catalog parsing fast on any input (the "never hang"
# guarantee). A real catalog is a tiny fraction of MAX_PRIMARY_CATALOG, so the
# cap never truncates one; it only bounds a marker-less/garbage region. Re-locks
# past corruption are capped at MAX_RESYNCS so a pathological region terminates.
MAX_PRIMARY_CATALOG = 2 * 1024 * 1024
MAX_RESYNCS = 256
# The backup-catalog signature also occurs by chance inside compressed data, so
# an image can contain many false hits; validating each one is not free (it may
# brute-force the seed). Only the newest few are checked - the real backup
# catalog is the last valid one - so the scan is bounded no matter how many
# chance hits (or an adversarial signature-filled region) are present.
MAX_BACKUP_CANDIDATES = 64


def parse_dir_fibs(byte_list, header_length='20', stop_on_garbage=False,
                   resync=False):
    """
    Parse the directory FIBs (catalog entries) from a decrypted byte buffer.

    Parsing stops at the first entry that can neither be decoded nor validated
    and from which the stream cannot re-lock - i.e. the end of the readable
    catalog. This bounds the work (a marker-less or non-QB image hands this a
    huge garbage region; the original code rescanned it entry by entry - the
    hang) and avoids emitting garbage entries past the real catalog.

    With resync=True, a corrupt entry mid-catalog (which would otherwise desync
    the variable-length parse and turn every following entry to garbage) is
    skipped by scanning forward to the next point where parsing re-locks, so
    entries after a bad sector are still recovered with their paths. Re-locking
    is capped at MAX_RESYNCS. (stop_on_garbage is retained for call
    compatibility; stopping at unrecoverable garbage is now always the
    behaviour.)
    """
    offset = 222  # catalog starts right after the 222-byte first-cylinder header
    dir_fibs = []
    resyncs = 0
    while offset < len(byte_list):
        try:
            dir_fib, new_offset = DirFib.from_bytes(
                byte_list, offset, header_length)
            ok = _entry_looks_valid(dir_fib)
        except (ValueError, struct.error, IndexError):
            dir_fib, new_offset, ok = None, None, False
        if not ok:
            # Keep a structurally-sound entry whose ONLY defect is an
            # out-of-range datestamp, as long as the NEXT entry still lines up.
            # A single bad datestamp field is real corruption that desyncs
            # nothing (the entry's own length is intact), so the file is
            # recovered with its correct path and flagged suspect-datestamp -
            # without loosening the resync guard, which still fires whenever the
            # name is garbage or the successor doesn't line up (a true desync).
            if (dir_fib is not None and _name_looks_valid(dir_fib)
                    and DirFib._next_entry_plausible(
                        byte_list, new_offset, header_length)):
                dir_fib.df_suspect_date = True
                dir_fibs.append(dir_fib)
                offset = new_offset
                continue
            # Genuine corruption: try to re-lock the stream past it. If we can't
            # (or have re-locked too many times), the readable catalog is over -
            # stop rather than emit garbage or grind the rest entry by entry.
            if resync and resyncs < MAX_RESYNCS:
                resyncs += 1
                rp = _find_catalog_resync(byte_list, offset + 1, header_length)
                if rp is not None:
                    offset = rp
                    continue
            break
        dir_fibs.append(dir_fib)
        offset = new_offset  # advance to the next entry

    return dir_fibs


def _score_header_length(byte_list, header_length, sample=16):
    """
    Parse up to `sample` catalog entries with the given layout and count how
    many look valid. The wrong layout desyncs almost immediately and produces
    out-of-range datestamps, so the correct layout scores far higher. The
    datestamp ranges are the discriminator (Amiga DateStamp: days since 1978,
    minutes < 1440, ticks < 3000 = 50/sec * 60).
    """
    offset = 222
    score = 0
    for _ in range(sample):
        try:
            fib, offset = DirFib.from_bytes(byte_list, offset, header_length)
        except Exception:
            break
        if _entry_looks_valid(fib):
            score += 1
    return score


def detect_header_length(byte_list):
    """
    Auto-detect whether catalog entries carry one file size (16-byte layout,
    older backups) or two (20-byte layout, newer). Parses the catalog both ways
    and returns the layout that yields more valid entries; ties favour '20'.
    """
    score20 = _score_header_length(byte_list, '20')
    score16 = _score_header_length(byte_list, '16')
    logging.debug("header-length detection scores: 20->%s 16->%s",
                  score20, score16)
    return '16' if score16 > score20 else '20'


def _seed_score(region_head, seed):
    """How many catalog entries decrypt plausibly under `seed` (best of either
    entry layout). Used to pick/recover the catalog encryption seed."""
    dec = decrypt_data(region_head, seed)
    return max(_score_header_length(dec, '20'), _score_header_length(dec, '16'))


def recover_seed(region, prefer_seed, sample=4096, good_score=2):
    """
    Resolve the catalog encryption seed. The seed is normally the low byte of
    the header time field (offset 0x0D); if that byte sits on a bad sector the
    whole catalog decrypts to noise. We score the on-disk seed first and, only
    if it parses poorly, brute-force all 256 seeds and take the best (the cipher
    is a stateless byte map, so 256 tries on a small prefix is cheap).

    Returns (seed, score, brute_forced).
    """
    head = region[:sample]
    prefer_seed = int(prefer_seed)
    base = _seed_score(head, prefer_seed)
    if base >= good_score:
        return prefer_seed, base, False
    best_seed, best = prefer_seed, base
    for s in range(256):
        sc = _seed_score(head, s)
        if sc > best:
            best, best_seed = sc, s
    return best_seed, best, best_seed != prefer_seed


def decrypt_byte(byte, encrypt_val):
    """Decrypt a single byte using the decryption table and the encryption value."""
    # Convert byte and encrypt_val to integers
    byte = int(byte)
    encrypt_val = int(encrypt_val)

    # Perform the operation with 8-bit wrap-around
    result = (byte - encrypt_val) & 0xFF

    # If result is 0, which can be a valid index, skip the check since index 0
    # is valid
    if result < 0 or result > 255:
        print(f"Invalid result: {result}")
        sys.exit()

    # Use the result to index into the decrypt_table
    return decrypt_table[result]


def decrypt_data(data, encrypt_val):
    """
    Decrypt a byte buffer with Quarterback's stateless substitution cipher.

    The original is `decryptTable[(x - encryptVal) & 0xFF]` per byte (Monitor.c),
    with no chained state, so the whole buffer decrypts as one vectorized numpy
    gather - byte-for-byte identical to decrypt_byte() but orders of magnitude
    faster, which is what keeps a large (e.g. marker-less) catalog region from
    feeling like a hang. Returns bytes; every caller treats it as a byte buffer.
    """
    # np.asarray(bytes, uint8) misparses a bytes object on numpy 2.x (it treats
    # it as one numeric string), so route bytes-like inputs through frombuffer;
    # ndarray/list inputs go through asarray. Both yield a uint8 buffer.
    if isinstance(data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(data, dtype=np.uint8)
    else:
        arr = np.asarray(data, dtype=np.uint8)
    idx = (arr.astype(np.int16) - int(encrypt_val)) & 0xFF
    return DECRYPT_TABLE[idx].tobytes()


def generate_path(path_stack, filename):
    """
    Join the paths from the path stack and filename using pathlib, which handles separators automatically.
    """
    # Construct the path using pathlib.Path
    path = Path(*[p[0] for p in path_stack]) / filename
    # Convert to string if needed for compatibility with other parts of the code
    return str(path)


def process_dirfibs(dirfibs):
    """
    Walk the catalog entries, building each one's full output path and creating
    directories. Each path component is sanitized individually; an unnamed or
    unsanitizable component is replaced with a placeholder so the directory
    structure (and the files beneath it) is preserved instead of lost.

    The path stack holds (component, remaining_child_count) pairs. Every entry
    decrements its parent's remaining count; directories also push a new level.
    """
    path_stack = [(DEFAULT_PATH, 0xBADCAFE)]
    placeholder_n = 0

    for dir_fib in dirfibs:
        if path_stack[-1][1] != 0xBADCAFE:
            # Subtract one from the current level's remaining child count for
            # every entry except the root.
            path_stack[-1] = (path_stack[-1][0], path_stack[-1][1] - 1)

        flags = dir_fib.df_flags
        if is_directory(flags):
            placeholder_n += 1
            comp = safe_component(dir_fib.df_name, f"_unnamed_dir_{placeholder_n:04d}")
            path_stack.append((comp, dir_fib.df_filcnt))
            dir_path = generate_path(path_stack, '')
            dir_fib.df_name = dir_path

            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {dir_path}: {e}; resetting to root.")
                path_stack = [(DEFAULT_PATH, 0xBADCAFE)]

        elif flags & FLAG_SEL_MASK:  # selected file (or link - it has no data)
            placeholder_n += 1
            comp = safe_component(dir_fib.df_name, f"_unnamed_file_{placeholder_n:04d}")
            dir_fib.df_name = generate_path(path_stack, comp)

        else:
            # Neither a directory nor a selected file: unexpected (corruption).
            # Don't tear down the whole tree - give it a placeholder path at the
            # current level so any data it has can still be saved, and continue.
            placeholder_n += 1
            comp = safe_component(dir_fib.df_name, f"_unknown_{placeholder_n:04d}")
            dir_fib.df_name = generate_path(path_stack, comp)

        while path_stack and path_stack[-1][1] == 0:
            path_stack.pop()

    return dirfibs


def _write_file(path, content, dir_fib=None):
    """
    Write `content` to `path`, creating parent directories. If `dir_fib` is
    given, stamp the file with its catalogued modification time. Returns True
    on success. Errors are reported and swallowed so one bad file never aborts
    the run.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(content)
    except OSError as e:
        print(f"Error saving file: {path}")
        print(e)
        return False
    if dir_fib is not None:
        # Stamp the catalogued modification time, but a corrupt/out-of-range
        # datestamp (e.g. a kept suspect-datestamp entry) must never fail the
        # file or abort the run - just leave it unstamped.
        try:
            ts = convert_to_unix_timestamp(
                dir_fib.df_days, dir_fib.df_minutes, dir_fib.df_ticks)
            os.utime(path, (ts, ts))
        except (OSError, ValueError, OverflowError):
            pass
    return True


def _align_markers_to_catalog(catalog_files, markers, matches):
    """
    Find the offset S that best lines the marker sequence up against the
    catalog (markers[k] <-> catalog_files[S+k]).

    Backups write file data in catalog order, so whichever disks are present
    contribute a contiguous run of the catalog's files - but that run may start
    anywhere (the first disk -> a prefix, the last disk -> a suffix, etc.).
    A single file's name+size is not unique (e.g. many 'Core.txt' of the same
    length), so we score the whole sequence: the correct offset matches nearly
    every marker, a wrong one only a coincidental few. Candidate offsets are the
    catalog positions whose file matches the first marker, keeping this cheap.
    """
    if not catalog_files or not markers:
        return 0
    candidates = [s for s in range(len(catalog_files))
                  if matches(catalog_files[s], markers[0])] or [0]
    best_s, best_score = candidates[0], -1
    for s in candidates:
        score = sum(1 for k in range(len(markers))
                    if s + k < len(catalog_files)
                    and matches(catalog_files[s + k], markers[k]))
        if score > best_score:
            best_score, best_s = score, s
    return best_s


def match_and_save_files(dir_fibs, file_list, default_path=DEFAULT_PATH,
                         suspect_regions=None):
    """
    Match data markers to catalog file entries and write each file to its
    catalogued path. Returns a per-file record list for the recovery report.

    The markers present are a contiguous run of the catalog (data is stored in
    catalog order), so we align the marker sequence to the catalog by offset and
    save markers[k] to catalog_files[S+k]'s path. This replaces the old +/-2
    positional window, which drifted badly whenever the marker count differed
    from the catalog file count (e.g. a partial/last-disk-only recovery) and
    could mis-assign duplicate-named files. Markers that don't line up are
    written to an '_unmatched' folder under unique names (rather than clobbering
    each other in a flat dump).
    """
    try:
        os.makedirs(default_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {default_path}")
        print(e)

    suspect_regions = suspect_regions or []
    records = []
    catalog_files = [f for f in dir_fibs if not is_directory(f.df_flags)]

    # The catalog basename was sanitized when its path was built (e.g. the
    # reserved name 'AUX' -> 'AUX_'), but the marker carries the raw name, so a
    # raw-vs-sanitized comparison would never match those. Sanitize the marker
    # name the same way before comparing. Cached: matches() is called many times
    # during the alignment search.
    _san_cache = {}

    def _san(name):
        v = _san_cache.get(name)
        if v is None:
            try:
                v = sanitize_filename(name).strip()
            except (ValueError, TypeError):
                v = ''
            _san_cache[name] = v
        return v

    def matches(cf, marker):
        return (Path(cf.df_name).name == _san(marker[1])
                and cf.df_size1 == marker[2])

    def status_for(marker):
        off, size = marker[0], marker[2]
        endmark = marker[8] if len(marker) > 8 else None
        if endmark is not None and endmark != size:
            return "qb-truncated"     # QB wrote fewer bytes than catalogued
        if overlaps_any(off, off + 40 + size, suspect_regions):
            return "suspect"
        if str(marker[5]).startswith("Bad size"):
            return "short"
        return "ok"

    def record_status(cf, marker):
        # A catalog entry kept despite an out-of-range datestamp is flagged so
        # the user can audit it - unless the marker's data has a worse issue.
        base = status_for(marker)
        if base == "ok" and getattr(cf, "df_suspect_date", False):
            return "suspect-datestamp"
        return base

    start = _align_markers_to_catalog(catalog_files, file_list, matches)

    prog = Progress("Writing files", total=len(file_list))

    # Walk catalog and markers together from the aligned start. On a local
    # discrepancy (a catalog entry whose data isn't on these disks, or an extra
    # marker), step forward only when a match confirms the re-alignment within a
    # small window - so a single gap doesn't abandon the rest, and duplicate
    # name+size files can't trigger a false jump.
    N, M, WINDOW = len(catalog_files), len(file_list), 8
    saved = 0
    matched = set()
    used_cat = set()
    ci, mi = start, 0
    while mi < M:
        if ci < N and matches(catalog_files[ci], file_list[mi]):
            if _write_file(catalog_files[ci].df_name, file_list[mi][6],
                           catalog_files[ci]):
                saved += 1
                records.append({'status': record_status(catalog_files[ci], file_list[mi]),
                                'size': file_list[mi][2],
                                'path': catalog_files[ci].df_name})
                prog.update()
                matched.add(mi)   # only on success: a failed write falls through
                                  # to the orphan pass -> _unmatched, never lost
            used_cat.add(ci)      # consumed either way (don't re-try a bad path)
            ci += 1
            mi += 1
            continue
        skip_cat = next((j for j in range(1, WINDOW + 1)
                         if ci + j < N
                         and matches(catalog_files[ci + j], file_list[mi])), None)
        skip_mark = next((j for j in range(1, WINDOW + 1)
                          if ci < N and mi + j < M
                          and matches(catalog_files[ci], file_list[mi + j])), None)
        if skip_cat is not None and (skip_mark is None or skip_cat <= skip_mark):
            ci += skip_cat       # catalog entries with no data on these disks
        elif skip_mark is not None:
            mi += skip_mark      # extra markers with no catalog slot (orphans)
        else:
            mi += 1              # lone unmatchable marker (orphan)

    # Name+size fallback. Quarterback doesn't always write file data in catalog
    # (pre-order) sequence, so the positional walk can orphan a long run of
    # files whose catalog slots genuinely exist, just out of order. Pair each
    # still-unmatched marker with an as-yet-unused catalog entry of the same
    # (sanitized name, size). Slots are kept in catalog order so duplicate-named
    # files fill sensibly; an assignment among true duplicates is flagged
    # 'rescued-ambiguous' since the directory can't be proven from name+size.
    remaining = {}
    for idx, cf in enumerate(catalog_files):
        if idx not in used_cat:
            remaining.setdefault((Path(cf.df_name).name, cf.df_size1), []).append(idx)
    ambiguous = {k for k, v in remaining.items() if len(v) > 1}

    rescued = 0
    for mi2, marker in enumerate(file_list):
        if mi2 in matched:
            continue
        slots = remaining.get((_san(marker[1]), marker[2]))
        if not slots:
            continue
        cf = catalog_files[slots.pop(0)]
        if _write_file(cf.df_name, marker[6], cf):
            rescued += 1
            matched.add(mi2)
            # This placement is itself uncertain (chosen by name+size only), so
            # always carry the rescued label; if the entry/marker ALSO has a
            # datestamp or data issue, append it rather than letting one mask the
            # other - every uncertain dimension stays visible in the report.
            kind = ("rescued-ambiguous"
                    if (_san(marker[1]), marker[2]) in ambiguous
                    else "rescued-namesize")
            base = record_status(cf, marker)
            status = kind if base == "ok" else f"{kind}+{base}"
            records.append({'status': status, 'size': marker[2],
                            'path': cf.df_name})
            prog.update()

    # Anything still unmatched: write under a unique name in _unmatched/ so its
    # data is never lost, even though it can't be placed in the directory tree.
    orphan = 0
    for k, marker in enumerate(file_list):
        if k not in matched:
            dest = str(Path(default_path) / '_unmatched' /
                       sanitize_filepath(f"{k:05d}_{marker[1]}"))
            if _write_file(dest, marker[6]):
                orphan += 1
                records.append({'status': 'unmatched', 'size': marker[2],
                                'path': dest})
                prog.update()

    prog.done()
    print(f"\nMatched {saved + rescued} of {len(file_list)} data markers to "
          f"catalog paths (catalog lists {len(catalog_files)} files"
          + (f"; {rescued} via name+size fallback" if rescued else "") + ").")
    if orphan:
        print(f"{orphan} unmatched data marker(s) written to "
              f"{Path(default_path) / '_unmatched'}/.")
    return records


def _unique_path(directory, name, used):
    """
    Return a path under `directory` for `name` that does not collide with one
    already used this run. The same filename recurs in many directories, so on
    a collision a numeric suffix is inserted before the extension
    (e.g. Core.txt -> Core_0001.txt). `used` (a set) is updated. This guarantees
    no file is silently overwritten when there is no catalog to give each its
    own directory.
    """
    name = sanitize_filepath(name) or "unnamed"
    candidate = str(Path(directory) / name)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = Path(name).suffix
    root = name[:len(name) - len(suffix)] if suffix else name
    i = 1
    while True:
        candidate = str(Path(directory) / f"{root}_{i:04d}{suffix}")
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def process_file_markers(file_list, default_path=DEFAULT_PATH, suspect_regions=None):
    """
    Recover files using only the data markers, with no catalog at all. Returns
    a per-file record list for the recovery report.

    Each marker carries a file's name and its data; only the directory tree
    lives in the catalog. So without a catalog we save every file by its name
    in one folder, making duplicate names unique so nothing is overwritten -
    no data is lost just because the catalog is missing or corrupt. Files are
    written in backup order, so the numeric suffixes on duplicates also reflect
    that order.
    """
    try:
        os.makedirs(default_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {default_path}")
        print(e)

    suspect_regions = suspect_regions or []
    used = set()
    records = []
    saved = 0
    prog = Progress("Writing files (markers only)", total=len(file_list))
    for marker in file_list:
        prog.update()
        target = _unique_path(default_path, marker[1], used)
        if _write_file(target, marker[6]):
            saved += 1
            off, size = marker[0], marker[2]
            endmark = marker[8] if len(marker) > 8 else None
            if endmark is not None and endmark != size:
                status = "qb-truncated"
            elif overlaps_any(off, off + 40 + size, suspect_regions):
                status = "suspect"
            elif str(marker[5]).startswith("Bad size"):
                status = "short"
            else:
                status = "ok"
            records.append({'status': status, 'size': size, 'path': target})

    prog.done()
    print(f"\nRecovered {saved} of {len(file_list)} files from markers alone "
          f"(no catalog); saved by filename in {default_path}/.")
    return records


def convert_to_unix_timestamp(df_days, df_minutes, df_ticks):
    # Define the start date as January 1, 1978
    start_date = datetime(1978, 1, 1)

    # Calculate the date based on the number of days since the start date
    file_date = start_date + timedelta(days=df_days)

    # Calculate the time based on minutes and ticks
    file_time = timedelta(minutes=df_minutes, seconds=df_ticks / 50.0)

    # Combine date and time
    full_datetime = file_date + file_time

    # Convert to Unix timestamp
    return full_datetime.timestamp()


def set_directory_timestamps(dirfibs):
    """
    Sets timestamps for each directory entry in dirfibs list.
    """
    for dir_fib in dirfibs:
        # Check if this entry is a directory
        if is_directory(dir_fib.df_flags):
            clean_path = dir_fib.df_name

            # Check if the directory exists before attempting to set timestamps
            if os.path.exists(clean_path):
                try:
                    # Convert to a Unix timestamp based on DirFib attributes
                    timestamp = convert_to_unix_timestamp(
                        dir_fib.df_days, dir_fib.df_minutes, dir_fib.df_ticks
                    )

                    # Apply the timestamp to the directory (both access and modification times)
                    os.utime(clean_path, (timestamp, timestamp))
                    # print(f"Timestamps set for directory: {clean_path}")

                except (OSError, ValueError, OverflowError) as e:
                    print(
                        f"Error setting timestamps for directory: {clean_path}")
                    print(e)
            else:
                print(
                    f"Directory does not exist, skipping timestamp update: {clean_path}")

    return dirfibs


def get_code(buf, pos, code_size):
    """
    Extracts a code from a buffer based on the given position and code size.
    Args:
        buf (bytes): The buffer containing the data.
        pos (int): The starting position of the code in bits.
        code_size (int): The size of the code in bits.
    Returns:
        int: The extracted code.
    """
    # Define masks based on code size
    masks = {
        9: 0x1ff,
        10: 0x3ff,
        11: 0x7ff,
        12: 0xfff,
        13: 0x1fff,
        14: 0x3fff,
        15: 0x7fff,
        16: 0xffff
    }
    mask = masks.get(code_size, 0xDEADBEEF)

    if mask == 0xDEADBEEF:
        print("Warning: Either code_size not set or not 9-16!!")
        return mask

    byte_pos = pos // 8
    bit_pos = pos % 8

    try:
        # Safely extract bytes and convert to an integer
        data = int.from_bytes(buf[byte_pos:byte_pos + 3], 'little')
        val = (data >> bit_pos) & mask
    except IndexError:
        # Fallback in case fewer bytes are available
        data = int.from_bytes(buf[byte_pos:byte_pos + 2], 'little')
        val = (data >> bit_pos) & mask

    logging.debug("pos: %s code_size: %s val: %s", pos, code_size, val)
    return val


def uncompress_me(buf, max_bits=MAX_COMPBIT):
    """
    This function performs the lzw decompression and makes up most of the
    complexity of the overall script. It takes a raw LZW datastream in
    as byte buffer, converts to a stream of bits, and then extracts
    variable-sized codes, and decompresses them.

    max_bits is the maximum LZW code size (9-16) taken from the file's
    compression flag; the code size never grows past it.

    See https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
    https://rosettacode.org/wiki/LZW_compression
    """
    # Read codes from a bytes copy of the datastream: get_code does an
    # int.from_bytes per code, which is ~2x faster on bytes than on a numpy
    # slice. The bit length is simply len*8 (the old np.unpackbits result was
    # used only for that length).
    n_bits = len(buf) * 8
    bbuf = bytes(buf)

    next_code = 258
    out = bytearray()          # decompressed bytes, assembled directly
    my_string = b""

    code_size = 9

    # 2^9 - 1
    maximum_table_size = 511

    # Building and initializing the dictionary: code x maps to the single byte
    # x. Dictionary values (and my_string) are bytes throughout, so the output
    # is built with no intermediate text/hex conversion.
    dictionary_size = 256
    dictionary = {x: bytes([x]) for x in range(dictionary_size)}

    # LZW Decompression algorithm

    i = 0

    while (i < (n_bits - 9)):

        # Don't allow code_size to grow past the file's declared max_bits.
        if next_code > maximum_table_size and code_size != max_bits:

            # need to bump the bit_pos by 9, essentially skipping a 9-bit word
            # Also bump code size by 1, and recalculate the maximum code
            # Only skip the 9-10 transition? not others? OK!
            if code_size == 9:
                i += code_size

            code_size += 1
            maximum_table_size = pow(2, int(code_size)) - 1

            logging.debug("Code size change at bits: %s", i)
            logging.debug("code_size is %s", code_size)
            logging.debug("Approximate decompress size is %s", len(out))

        code = get_code(bbuf, i, code_size)

        if code == 0xDEADBEEF:
            return str.encode("DEADBEEF")

        if code > next_code:
            logging.debug(
                "Code read is higher than it should be. code is %s next_code is %s",
                code,
                next_code)

        if code not in dictionary:
            # KwKwK case. If my_string is empty the stream is invalid or
            # misaligned (e.g. a corrupt file, or a wrong-width trial decode):
            # stop and keep whatever decoded cleanly so far rather than crash.
            if not my_string:
                break
            # my_string[0:1] (a 1-byte bytes), NOT my_string[0] (an int).
            dictionary[code] = my_string + my_string[0:1]

        out += dictionary[code]

        if len(my_string) != 0:
            dictionary[next_code] = my_string + dictionary[code][0:1]
            next_code += 1
            logging.debug("next_code is now %s", next_code)

        my_string = dictionary[code]

        i += code_size

    return bytes(out)


def load_file(filepath):
    """Loads the binary file into a numpy array."""
    return np.fromfile(filepath, dtype=np.ubyte)


DISK_SIZE = 901120  # bytes in a standard Amiga DD floppy / one backup disk


def natural_key(name):
    """Sort key so '2.adf' sorts before '10.adf' (not lexically)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]


def expand_inputs(paths):
    """
    Expand the given paths into a flat list of files. A directory is expanded
    to the .adf files it contains, naturally sorted by name.
    """
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            adfs = sorted(
                (str(x) for x in path.iterdir()
                 if x.is_file() and x.suffix.lower() == '.adf'),
                key=lambda s: natural_key(Path(s).name))
            if not adfs:
                print(f"Warning: no .adf files found in directory {p}")
            files.extend(adfs)
        else:
            files.append(str(path))
    return files


def load_inputs(paths):
    """
    Loads one or more backup inputs and concatenates them into a single image
    so the user no longer has to 'cat' the disks together by hand. Accepts
    individual disk files and/or directories of .adf files.

    When several raw single-disk (901120-byte) images are given, they are
    ordered by the diskNum field in each disk's header -- the authoritative
    sequence number -- rather than trusting the order they were passed in or
    the shell's lexical glob order. A single, already-combined file is loaded
    as-is. The resolved order is printed, with warnings for missing or
    duplicated disk numbers.
    """
    files = expand_inputs(paths)
    if not files:
        print("Error: no input files to process.")
        sys.exit(1)

    loaded = []
    for f in files:
        try:
            arr = load_file(f)
        except OSError as e:
            print(f"Error opening backup file: {f}")
            print(e)
            continue
        # The header's date+time (offsets 6-13, unencrypted) is stamped
        # identically on every disk of one backup, so it identifies the set.
        stamp = (struct.unpack_from('>II', arr.tobytes(), 6)
                 if len(arr) >= 14 else None)
        loaded.append({
            'path': f,
            'arr': arr,
            'size': len(arr),
            'diskNum': int(arr[4]) if len(arr) >= 5 else None,
            'seq': arr[2:4].tobytes().decode('latin-1') if len(arr) >= 4 else '',
            'stamp': stamp,
        })

    if not loaded:
        print("Error: none of the input files could be opened.")
        sys.exit(1)

    meta = {
        'num_files': len(loaded),
        'warnings': [],
        'order_basis': 'single file',
        'disks': [],
    }

    if len(loaded) == 1:
        meta['combined_size'] = loaded[0]['size']
        meta['disks'] = [(Path(loaded[0]['path']).name,
                          loaded[0]['diskNum'], loaded[0]['size'])]
        return loaded[0]['arr'], meta

    # Order multiple raw disks by their header sequence number when possible.
    all_raw_disks = all(x['size'] == DISK_SIZE for x in loaded)
    have_disknums = all(x['diskNum'] not in (None, 0) for x in loaded)
    if all_raw_disks and have_disknums:
        loaded.sort(key=lambda x: x['diskNum'])
        meta['order_basis'] = "disk header sequence number"
    else:
        meta['order_basis'] = "filename order"

    nums = []
    for x in loaded:
        meta['disks'].append((Path(x['path']).name, x['diskNum'], x['size']))
        if x['size'] == DISK_SIZE:
            nums.append(x['diskNum'])

    if nums:
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        if dupes:
            meta['warnings'].append(
                f"duplicate disk numbers {dupes} - order/contents may be wrong")
        missing = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
        if missing:
            meta['warnings'].append(
                f"missing disk numbers {missing} - set is incomplete; files "
                "spanning a gap may not fully restore")

    stamps = {x['stamp'] for x in loaded if x['size'] == DISK_SIZE and x['stamp']}
    if len(stamps) > 1:
        meta['warnings'].append(
            "disks carry different backup date/time stamps - they may be from "
            "different backups, not one set")

    combined = np.concatenate([x['arr'] for x in loaded])
    meta['combined_size'] = len(combined)
    return combined, meta


def detect_multidisk(full_file):
    """
    Reassemble a combined multi-disk 'Qb' image into one contiguous logical
    stream.

    Disk 1 keeps its full header (the catalog lives there, at offset 222).
    Every later disk is prefixed by a 16-byte continuation header, which is
    removed so that file data spanning a disk boundary becomes contiguous and
    reassembles automatically: the modern format has no CONT markers, so a file
    is simply its df_Size bytes in the logical stream regardless of where the
    physical disk split falls.
    """
    meta = {'multidisk': False, 'disks_reassembled': 1, 'warnings': []}
    if len(full_file) <= DISK_SIZE:
        return full_file, meta

    cuts = []
    for boundary in range(DISK_SIZE, len(full_file), DISK_SIZE):
        if bytes(full_file[boundary:boundary + 2]) == QB_MODERN_ID:
            cuts.append((boundary, boundary + MODERN_CONT_HEADER))
        else:
            meta['warnings'].append(
                f"no 'Qb' continuation header at offset {boundary} - "
                "disk order or contents may be wrong")

    if cuts:
        meta['multidisk'] = True
        meta['disks_reassembled'] = len(cuts) + 1
        full_file = np.delete(
            full_file, np.concatenate([np.arange(s, e) for s, e in cuts]))
    return full_file, meta


def find_markers(full_file):
    """
    Finds data markers in the binary file.

    There are four marker tags, all sharing the same 40-byte delimiter layout
    (tag + 32-byte name + 4-byte size):
      'FMRK'  uncompressed file
      'IMRK'  uncompressed disk image
      'CFM'   compressed file   (3-byte tag + 1 compression-flag byte)
      'CIM'   compressed disk image (3-byte tag + 1 compression-flag byte)

    For the compressed tags the flag byte's low 6 bits (COMP_BIT_MASK 0x3F) are
    the LZW maximum code size and the 0x80 bit (COMP_ON_FLAG) signals
    compression on. The original code matched only 'CFM' with flag 0x90 (on +
    16-bit); we now accept any valid width 9-16, record it per marker, and also
    recognise the disk-image variants ('IMRK'/'CIM').
    """
    offset_list = []

    # Uncompressed markers: exact 4-byte tag.
    for tag, sig in (('FMRK', b'FMRK'), ('IMRK', b'IMRK')):
        for p in np.where(full_file == sig[0])[0]:
            if bytes(full_file[p:p + 4]) == sig:
                offset_list.append({'tag': tag, 'offset': int(p), 'max_bits': 0})

    # Compressed markers: 3-byte tag + validated compression-flag byte (so we
    # don't match "CFM"/"CIM" byte sequences occurring inside file data).
    for tag, sig in (('CFM', b'CFM'), ('CIM', b'CIM')):
        for p in np.where(full_file == sig[0])[0]:
            if bytes(full_file[p:p + 3]) == sig:
                flag = int(full_file[p + 3])
                max_bits = flag & COMP_BIT_MASK
                if (flag & COMP_ON_FLAG) and MIN_COMPBIT <= max_bits <= MAX_COMPBIT:
                    offset_list.append(
                        {'tag': tag, 'offset': int(p), 'max_bits': max_bits})

    # Recover uncompressed markers whose 4-byte tag has a single corrupt byte,
    # validated structurally so this adds nothing on a clean disk.
    exact = {e['offset'] for e in offset_list}
    offset_list.extend(find_fuzzy_markers(full_file, exact))

    # Sort by offset so downstream next-marker boundary detection is correct
    # even when a backup mixes compressed and uncompressed markers.
    offset_list.sort(key=lambda e: e['offset'])

    logging.debug(offset_list)
    return offset_list


def find_fuzzy_markers(full_file, exact_offsets):
    """
    Recover uncompressed file/image markers whose 4-byte tag is corrupted in
    one byte. A real delimiter is tag(4) + name(32, null-terminated, printable)
    + size(4), so we anchor on positions where 3 of the 4 tag bytes still match
    'FMRK'/'IMRK' and then require the name and size fields to be structurally
    valid. That combination is specific enough that clean data yields no false
    positives, while a single flipped tag byte is still recovered.
    """
    found = []
    n = len(full_file)
    if n < 40:
        return found
    for tag, name in ((b'FMRK', 'FMRK'), (b'IMRK', 'IMRK')):
        sig = np.frombuffer(tag, np.uint8)
        matches = np.zeros(n - 3, dtype=np.int16)
        for j in range(4):
            matches += (full_file[j:n - 3 + j] == sig[j])
        for p in np.where(matches == 3)[0]:   # exactly one tag byte wrong
            p = int(p)
            if any(abs(p - e) < 40 for e in exact_offsets):
                continue
            namate = bytes(full_file[p + 4:p + 36])
            nul = namate.find(0)
            if nul < 1:                         # need a non-empty, terminated name
                continue
            label = namate[:nul].decode('latin-1', 'replace')
            if not all(c.isprintable() for c in label):
                continue
            size = int.from_bytes(bytes(full_file[p + 36:p + 40]), 'big')
            if not 0 <= size <= MAX_SANE_FILE:
                continue
            found.append({'tag': name, 'offset': p, 'max_bits': 0, 'fuzzy': True})
    return found


# Marker tags grouped by how their data is stored.
COMPRESSED_TAGS = ('CFM', 'CIM')      # LZW-compressed payload
UNCOMPRESSED_TAGS = ('FMRK', 'IMRK')  # raw payload


def extract_file_info(full_file, offset_list):
    """Extracts file information based on found markers."""
    file_list = []

    for entry in offset_list:
        tag = entry['tag']
        offset = entry['offset']
        max_bits = entry['max_bits']

        filename = "".join(
            [chr(item) for item in full_file[offset + 4:offset + 34]]).split('\x00', 1)[0]

        filesize1_bits = np.unpackbits(full_file[offset + 36:offset + 40])
        filesize1 = int(filesize1_bits.dot(
            2 ** np.arange(filesize1_bits.size)[::-1]))

        # For uncompressed files, the 4-byte OutputEndMark right after the data
        # is the actual number of bytes Quarterback wrote. If it differs from
        # the catalogued size, QB itself couldn't read the whole file at backup
        # time - a useful "already damaged in the backup" signal. (index 8;
        # None for compressed files, whose data end isn't at +40+size.)
        endmark = None
        if tag in UNCOMPRESSED_TAGS:
            em_at = offset + 40 + filesize1
            if em_at + 4 <= len(full_file):
                endmark = int.from_bytes(bytes(full_file[em_at:em_at + 4]), 'big')

        # Trailing fields: [7] LZW max code size (0 for uncompressed),
        # [8] OutputEndMark real-size (None if N/A).
        file_list.append([offset, filename, filesize1, 0, tag, '', '',
                          max_bits, endmark])

    return file_list


def uncompress_data(full_file, file_list):
    """
    Decompresses compressed payloads (CFM files, CIM disk images) and copies
    uncompressed payloads (FMRK files, IMRK disk images). Image markers share
    the same delimiter and data encoding as their file counterparts, so they
    are handled identically here.
    """
    prog = Progress("Decompressing files", total=len(file_list))
    for j in range(len(file_list)):
        prog.update()
        if file_list[j][4] in COMPRESSED_TAGS:
            cfm_offset = file_list[j][0]
            # Proper path: use the max code size declared in the marker flag.
            max_bits = file_list[j][7] or MAX_COMPBIT
            end_lzw_datastream = determine_end_of_datastream(
                full_file, file_list, j)

            start_lzw_datastream = cfm_offset + 40
            expected = file_list[j][2]

            data = uncompress_me(
                full_file[start_lzw_datastream:end_lzw_datastream], max_bits)

            if len(data) == expected:
                file_list[j][6] = data
                file_list[j][5] = "Decompressed OK"
            else:
                # Heuristic: the boundary may be one byte short.
                data_retry = uncompress_me(
                    full_file[start_lzw_datastream:end_lzw_datastream + 1], max_bits)
                if len(data_retry) == expected:
                    file_list[j][6] = data_retry
                    file_list[j][5] = "Decompressed OK on +1 retry"
                else:
                    # Recovery layer: the marker's width byte may be corrupt or
                    # absent (e.g. marker found by a future structural scan).
                    # Trial-decompress at every plausible width and accept the
                    # one whose output length matches the expected file size.
                    recovered = try_all_bit_widths(
                        full_file, start_lzw_datastream, end_lzw_datastream,
                        expected, skip=max_bits)
                    if recovered is not None:
                        width, data_ok = recovered
                        file_list[j][6] = data_ok
                        file_list[j][5] = (
                            "Decompressed OK via width auto-detect "
                            "(%d-bit)" % width)
                    else:
                        # Keep the best-effort partial output; data before any
                        # corruption point is still valid. The shortfall is
                        # tallied for the run report rather than printed inline.
                        file_list[j][6] = data
                        file_list[j][5] = (
                            "Bad size: got %d of %d bytes" % (len(data), expected))

        elif file_list[j][4] in UNCOMPRESSED_TAGS:
            file_list[j][6] = full_file[file_list[j][0] +
                                        40:file_list[j][0] + file_list[j][2] + 40]

    prog.done()
    return file_list


def try_all_bit_widths(full_file, start, end, expected, skip=None):
    """
    Recovery fallback for compressed files whose declared LZW max code size is
    missing or corrupt. Decompresses the datastream at each candidate width and
    returns (width, data) for the first one whose decompressed length matches
    the expected (catalog/marker) file size, or None if none match.
    """
    for width in range(9, MAX_COMPBIT + 1):
        if width == skip:
            continue
        for end_candidate in (end, end + 1):
            data = uncompress_me(full_file[start:end_candidate], width)
            if len(data) == expected:
                return width, data
    return None


def determine_end_of_datastream(full_file, file_list, index):
    """Determines the end of the LZW datastream."""
    end_lzw_datastream = 0
    next_cfm_offset = 0

    if index == len(file_list) - 1:
        end_lzw_datastream = len(full_file)
    elif index < len(file_list):
        next_cfm_offset = file_list[index + 1][0]

    if end_lzw_datastream == 0:
        for i in range(4):
            candidate_bits = np.unpackbits(
                full_file[next_cfm_offset - 4 - i:next_cfm_offset - i])
            candidate_size = candidate_bits.dot(
                2 ** np.arange(candidate_bits.size)[::-1])

            if candidate_size == file_list[index][2]:
                end_lzw_datastream = next_cfm_offset - 4 - i
                break

    return end_lzw_datastream


def detect_format(full_file):
    """
    Identify the backup format from the first two bytes of disk 1: 'QB'
    (0x5142) is the antique V4.x format, anything else is treated as the
    modern v5.x/6.x 'Qb' format.
    """
    if len(full_file) >= 2 and bytes(full_file[0:2]) == QB_ANTIQUE_ID:
        return 'antique'
    return 'modern'


def strip_antique_headers(full_file):
    """
    For a combined antique set, remove the 14-byte 'QB' header that prefixes
    each disk after the first, so the concatenated file data is contiguous and
    can be sliced across disk boundaries. (Single disks pass through unchanged.)
    """
    if len(full_file) <= DISK_SIZE:
        return full_file
    cuts = []
    for boundary in range(DISK_SIZE, len(full_file), DISK_SIZE):
        if bytes(full_file[boundary:boundary + 2]) == QB_ANTIQUE_ID:
            cuts.append((boundary, boundary + ANTIQUE_HEADER_SIZE))
    if cuts:
        full_file = np.delete(
            full_file, np.concatenate([np.arange(s, e) for s, e in cuts]))
    return full_file


def parse_antique_catalog(full_file, base=0):
    """
    Parse the V4.x 'QB' catalog (unencrypted, immediately after the 14-byte
    disk header at `base`). Returns (entries, data_start):

      entries     a flat pre-order (data-order) list of dicts with keys
                  size/date/time/ticks/filcnt/flags/name/path/isdir
      data_start  byte offset where the concatenated file data begins

    Each entry is size(4) date(2) time(2) ticks(2) filcnt(2) prot(1) flags(1)
    followed by a null-terminated name and comment. The tree is walked using
    df_FilCnt (count-driven), which also tells us exactly where the catalog
    ends and file data starts. `base` is 0 for the primary catalog on disk 1
    and the 'QBc2' offset for the backup catalog on the last disk.
    """
    n = len(full_file)
    root_count = int.from_bytes(bytes(full_file[base + 14:base + 16]), 'big')
    entries = []
    state = {'off': base + 16}

    def read_cstr(limit):
        off = state['off']
        end = off
        stop = min(n, off + limit + 1)
        while end < stop and full_file[end] != 0:
            end += 1
        raw = bytes(full_file[off:end])
        # Skip the terminating null (the assembler reader does the same even
        # when a name fills its field exactly).
        state['off'] = end + 1 if end < n else n
        return raw

    def read_level(count, path):
        for _ in range(count):
            off = state['off']
            if off + ANTIQUE_ENTRY_HEADER > n:
                return False
            size = int.from_bytes(bytes(full_file[off:off + 4]), 'big')
            date = int.from_bytes(bytes(full_file[off + 4:off + 6]), 'big')
            time = int.from_bytes(bytes(full_file[off + 6:off + 8]), 'big')
            ticks = int.from_bytes(bytes(full_file[off + 8:off + 10]), 'big')
            filcnt = int.from_bytes(bytes(full_file[off + 10:off + 12]), 'big')
            flags = int(full_file[off + 13])
            state['off'] = off + ANTIQUE_ENTRY_HEADER
            name = read_cstr(ANTIQUE_NAME_MAX).decode('iso-8859-1', errors='replace')
            read_cstr(ANTIQUE_NOTE_MAX)  # comment - not used on restore
            isdir = bool(flags & ANTIQUE_FLAG_DIR)
            full_path = sanitize_filepath(str(Path(path) / name)) if name else path
            entries.append({
                'size': size, 'date': date, 'time': time, 'ticks': ticks,
                'filcnt': filcnt, 'flags': flags, 'name': name,
                'path': full_path, 'isdir': isdir,
            })
            if isdir and filcnt > 0:
                if not read_level(filcnt, full_path):
                    return False
        return True

    read_level(root_count, DEFAULT_PATH)
    return entries, state['off']


# A single file larger than this is taken as a corrupt size field, not a real
# file (Amiga floppy-era files don't approach this; corrupt fields usually read
# as huge values with high bits set).
MAX_SANE_FILE = 32 * 1024 * 1024

# Recognizable Amiga file-start signatures, used to re-anchor the sequential
# read after a corrupt size. 4-byte tags only (low false-positive rate).
AMIGA_SIGNATURES = [
    (b'\x00\x00\x03\xf3', 'Amiga HUNK executable/library'),
    (b'FORM', 'IFF FORM'),
    (b'CAT ', 'IFF CAT'),
    (b'LIST', 'IFF LIST'),
]


def find_antique_alt_catalog(full_file):
    """Locate the antique backup catalog ('QB' + 'c2' = b'QBc2') on the last
    disk of a V4.x set. Returns its byte offset or None."""
    idx = full_file.tobytes().find(b'QBc2')
    return idx if idx != -1 else None


def find_next_file_signature(full_file, start, limit):
    """
    Scan [start, start+limit) for the next recognizable Amiga file-start
    signature and return its offset, or None. Used to re-anchor after a corrupt
    size has thrown off the sequential read; typed files (executables, IFF)
    re-anchor cleanly, packed text regions have no signpost.
    """
    end = min(len(full_file), start + limit)
    if end <= start:
        return None
    window = full_file[start:end].tobytes()
    best = None
    for sig, _ in AMIGA_SIGNATURES:
        i = window.find(sig)
        if i != -1 and (best is None or i < best):
            best = i
    return start + best if best is not None else None


def cross_validate_antique_sizes(primary, alt):
    """
    Compare the primary catalog's file sizes against the backup ('QBc2')
    catalog and repair an implausible primary size from the backup when the two
    entries clearly correspond (same name and position). Mutates `primary`
    entries in place; returns (repaired, disagreements) counts.
    """
    repaired = disagreements = 0
    for p, a in zip(primary, alt):
        if p['name'] != a['name'] or p['isdir'] != a['isdir']:
            break  # structures diverged; stop trusting the pairing
        if p['isdir'] or p['size'] == a['size']:
            continue
        disagreements += 1
        p_ok = 0 <= p['size'] <= MAX_SANE_FILE
        a_ok = 0 <= a['size'] <= MAX_SANE_FILE
        if not p_ok and a_ok:           # primary corrupt, backup sane -> repair
            p['size'] = a['size']
            p['size_repaired'] = True
            repaired += 1
    return repaired, disagreements


def extract_antique(full_file, args, set_meta=None):
    """
    Extract an antique V4.x 'QB' backup. The catalog is plaintext and there are
    no file markers: file data is concatenated after the catalog in catalog
    order, so we slice each file's df_Size bytes sequentially. Data that would
    live on a disk we don't have is reported as missing; a file straddling the
    end of the available data is written as a (valid) partial.
    """
    full_file = strip_antique_headers(full_file)
    hdr = parse_backup_header(full_file, 'antique')

    entries, data_start = parse_antique_catalog(full_file)
    dirs = [e for e in entries if e['isdir']]
    files = [e for e in entries if not e['isdir']]

    # #2: cross-validate sizes against the backup ('QBc2') catalog if present,
    # repairing an implausible primary size from the backup copy.
    repaired = disagreements = 0
    alt_off = find_antique_alt_catalog(full_file)
    if alt_off:
        try:
            alt_entries, _ = parse_antique_catalog(full_file, base=alt_off)
            alt_files = [e for e in alt_entries if not e['isdir']]
            repaired, disagreements = cross_validate_antique_sizes(files, alt_files)
        except Exception:  # noqa: BLE001 - a damaged backup catalog just isn't used
            pass

    # #1: catalog size fields that are implausibly large are corrupt; flag them.
    implausible = sum(1 for e in files if not (0 <= e['size'] <= MAX_SANE_FILE))

    flat = args.ignore_catalog
    catalog_desc = (f"plaintext, {len(files)} files, {len(dirs)} dirs; "
                    f"data at offset {data_start}")
    if flat:
        catalog_desc = "ignored - files written flat (catalog still locates data)"

    warnings = []
    if alt_off:
        warnings.append(f"backup catalog at offset {alt_off}: {repaired} "
                        f"size(s) repaired from it, {disagreements} disagreement(s)")
    if implausible:
        warnings.append(f"{implausible} catalog size(s) implausibly large "
                        "(corrupt); the sequential read re-anchors past them")

    meta = dict(set_meta or {'num_files': 1, 'disks': [],
                'combined_size': len(full_file), 'order_basis': 'single file'})
    meta['warnings'] = list(meta.get('warnings', [])) + warnings
    marker_meta = {'total': 0, 'by_tag': {'FMRK': 0, 'IMRK': 0, 'CFM': 0, 'CIM': 0},
                   'widths': []}
    report_text = render_report(__version__, meta, None, hdr, marker_meta,
                                catalog_desc)

    try:
        os.makedirs(DEFAULT_PATH, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {DEFAULT_PATH}")
        print(e)

    if not flat:
        for e in dirs:
            try:
                os.makedirs(e['path'], exist_ok=True)
            except OSError as err:
                print(f"Error creating directory: {e['path']}")
                print(err)

    def out_path_for(e):
        if flat:
            return str(Path(DEFAULT_PATH) / sanitize_filepath(e['name']))
        return e['path']

    def write_chunk(e, chunk):
        out_path = out_path_for(e)
        try:
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(chunk)
        except OSError as err:
            print(f"Error saving file: {out_path}")
            print(err)
            return False
        try:
            ts = convert_to_unix_timestamp(e['date'], e['time'], e['ticks'])
            os.utime(out_path, (ts, ts))
        except (OSError, ValueError, OverflowError):
            pass
        return True

    # Sequentially slice each file's data by its catalogued size. Note that a
    # corrupt *byte* does not desync this (positions come from the catalog, not
    # the data) - only a corrupt *size* does, which #3 recovers from.
    suspect_regions = detect_fill_runs(full_file)
    avail = len(full_file)
    pos = data_start
    saved = partial = missing = best_effort = reanchored = 0
    records = []
    prog = Progress("Writing files", total=len(files))
    for e in files:
        prog.update()
        size = e['size']
        plausible = 0 <= size <= MAX_SANE_FILE

        if pos >= avail:
            missing += 1
            if plausible:
                pos += size
            continue

        if not plausible:
            # #3: the size is corrupt, so pos += size would be meaningless.
            # Re-anchor to the next recognizable file start and write whatever
            # lies in between as the best-effort recovery of this file.
            nxt = find_next_file_signature(
                full_file, pos + 4, min(avail - pos, 8 * 1024 * 1024))
            end = nxt if nxt is not None else avail
            if write_chunk(e, bytes(full_file[pos:end])):
                best_effort += 1
                records.append({'status': 'best-effort', 'size': end - pos,
                                'path': out_path_for(e)})
            if nxt is not None:
                reanchored += 1
            print(f"Corrupt catalog size for {e['name']!r}; recovered "
                  f"{end - pos} bytes best-effort"
                  + (" and re-anchored to next file" if nxt is not None else ""))
            pos = end
            continue

        end = pos + size
        if write_chunk(e, bytes(full_file[pos:min(end, avail)])):
            saved += 1
            status = "suspect" if overlaps_any(pos, end, suspect_regions) else "ok"
            if end > avail:
                partial += 1
                status = "partial"
            records.append({'status': status, 'size': size,
                            'path': out_path_for(e)})
        pos = end
    prog.done()

    if not flat:
        for e in dirs:
            if os.path.exists(e['path']):
                try:
                    ts = convert_to_unix_timestamp(e['date'], e['time'], e['ticks'])
                    os.utime(e['path'], (ts, ts))
                except (OSError, ValueError, OverflowError):
                    pass

    msg = f"\nDone. Saved {saved} files"
    if partial:
        msg += f" ({partial} partial at a disk boundary)"
    if best_effort:
        msg += (f"; {best_effort} written best-effort past a corrupt catalog "
                f"size ({reanchored} re-anchored to the next file)")
    if missing:
        msg += f"; {missing} files had no data on the supplied disk(s)"
    print(msg + ".")

    summary = msg.strip()
    write_recovery_report(report_text, records, summary,
                          str(Path(DEFAULT_PATH) / "_recovery_report.txt"))


def _catalog_score_at(full_file, base, sample=8192):
    """
    Bounded test of whether a real catalog begins at byte offset `base`: recover
    the seed from this header's own seed byte (base+0x0D) over a small prefix and
    count how many of the first entries decrypt and parse as plausible DirFibs
    (entries start 222 bytes after the header). Returns that count (0 = no
    catalog here). Bounded, so it is cheap to call on every signature candidate.
    """
    if base + 0xD >= len(full_file):
        return 0
    region = full_file[base:base + min(len(full_file) - base, sample)]
    prefer = int(full_file[base + 0xD])
    _seed, score, _brute = recover_seed(region, prefer)
    return score


def find_backup_catalog(full_file):
    """
    Locate the backup (alternate) catalog, written on the LAST disk of a set and
    identified by a 'Qbc2' header (QB_CAT_ID) or the early-V5.0 'Qb\\0\\xC2'
    variant (QB_O_CAT_ID). Returns its byte offset, or None.

    The signature bytes also occur by chance inside compressed file data, so the
    first match is often a false positive that recovers nothing (the real backup
    catalog sits on the last disk, after all the file data). We therefore look at
    occurrences NEWEST-first and return the first that a bounded parse confirms
    is a real catalog. Only the last MAX_BACKUP_CANDIDATES occurrences are
    checked, so a region full of chance (or adversarial) signature hits still
    terminates quickly - validating each candidate may brute-force the seed.
    """
    raw = full_file.tobytes()
    offsets = []
    for sig in (b'Qbc2', bytes([0x51, 0x62, 0x00, 0xC2])):
        start = 0
        while True:
            idx = raw.find(sig, start)
            if idx == -1:
                break
            offsets.append(idx)
            start = idx + 1
    for idx in sorted(offsets, reverse=True)[:MAX_BACKUP_CANDIDATES]:
        if _catalog_score_at(full_file, idx) >= 2:   # validates as a catalog
            return idx                                # newest valid one wins
    return None


def _load_catalog_from(full_file, base, region):
    """
    Recover the encryption seed for the catalog at byte offset `base`, decrypt
    `region`, and parse it into DirFibs (building paths / creating directories).
    Returns (dir_fibs_or_None, header_length, seed, seed_score, brute_forced).
    """
    prefer_seed = int(full_file[base + 0xD]) if len(full_file) > base + 0xD else 0
    seed, seed_score, brute = recover_seed(region, prefer_seed)
    decrypted = decrypt_data(region, seed)
    dir_fibs, header_length = prepare_dir_fibs(decrypted)
    return dir_fibs, header_length, seed, seed_score, brute


def resolve_catalog(full_file, first_marker, ignore_catalog):
    """
    Decide which catalog (if any) drives extraction, trying sources from most to
    least reliable and never giving up early:

        primary (disk 1)  ->  validated backup (last disk)  ->  none (markers)

    The backup catalog is only consulted when the primary is unreadable, so a
    healthy backup pays nothing extra; when the primary IS unreadable, the
    secondary copy on the last disk is tried before falling back to recovering
    every file from its data marker. Returns (dir_fibs_or_None, hdr,
    catalog_desc, warnings); dir_fibs is None when the caller should recover
    from markers alone. No files are written here.
    """
    warnings = []
    if ignore_catalog:
        hdr = parse_backup_header(full_file, 'modern')
        return None, hdr, "ignored - recovering from data markers only", warnings

    log_phase("Resolving catalog (primary -> backup -> markers)...")

    # 1) Primary catalog on disk 1, bounded to the first data marker and capped.
    region = full_file[0:min(first_marker, MAX_PRIMARY_CATALOG)]
    dir_fibs, hl, seed, score, brute = _load_catalog_from(full_file, 0, region)
    source, base = "primary", 0

    # 2) Primary unreadable -> the validated backup catalog on the last disk.
    if not dir_fibs:
        backup_off = find_backup_catalog(full_file)
        if backup_off is not None:
            b_region = full_file[backup_off:backup_off
                                 + min(len(full_file) - backup_off, 262144)]
            b_fibs, b_hl, b_seed, b_score, b_brute = _load_catalog_from(
                full_file, backup_off, b_region)
            if b_fibs:
                dir_fibs, hl, seed, score, brute = (
                    b_fibs, b_hl, b_seed, b_score, b_brute)
                source, base = f"backup (offset {backup_off})", backup_off
                warnings.append(
                    "primary catalog unreadable; recovered from the backup "
                    f"catalog on the last disk (offset {backup_off})")

    # 3) Neither catalog usable -> caller recovers from data markers alone.
    if not dir_fibs:
        hdr = parse_backup_header(full_file, 'modern')
        return (None, hdr,
                "primary and backup catalogs unreadable - recovering from markers",
                warnings)

    if brute:
        warnings.append(
            f"catalog seed byte (0x0D) unreadable; recovered seed {seed:#04x} "
            f"by brute force ({score} entries parse)")

    hdr = parse_backup_header(full_file, 'modern', base=base, seed_override=seed)
    n_files = sum(not is_directory(f.df_flags) for f in dir_fibs)
    n_dirs = sum(is_directory(f.df_flags) for f in dir_fibs)
    n_links = sum(bool(f.df_flags & (FLAG_HLINK_MASK | FLAG_SLINK_MASK))
                  for f in dir_fibs)
    desc = (f"{source}, {hl}-byte entries - {n_files} files, {n_dirs} dirs, "
            f"{n_links} links")
    return dir_fibs, hdr, desc, warnings


def prepare_dir_fibs(decrypted_catalog, header_length_arg='auto'):
    """
    Resolve the entry layout, parse the DirFib tree, and build each entry's
    path (creating directories). Returns (dir_fibs, header_length); dir_fibs is
    empty/None if the catalog cannot be parsed, and the caller then tries the
    next recovery source. Parsing stops at the end of the readable catalog
    (parse_dir_fibs), so a corrupt catalog can't spawn junk directories.
    """
    header_length = (detect_header_length(decrypted_catalog)
                     if header_length_arg == 'auto' else header_length_arg)
    try:
        dir_fibs = parse_dir_fibs(decrypted_catalog, header_length=header_length,
                                  resync=True)
        dir_fibs = process_dirfibs(dir_fibs)
    except Exception as e:  # noqa: BLE001 - any catalog damage falls back to markers
        log_phase(f"  (catalog parse failed: {e})")
        return None, header_length
    return dir_fibs, header_length


def days_to_date(days):
    """Render an Amiga datestamp (days since 1978-01-01) as YYYY-MM-DD."""
    try:
        return (datetime(1978, 1, 1) + timedelta(days=int(days))).strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return "?"


def _printable(text):
    """Keep only printable characters of a header string (header tails on disks
    other than the first are file data and decode to noise)."""
    return ''.join(c for c in text if c.isprintable()).strip()


def parse_backup_header(full_file, fmt, base=0, seed_override=None):
    """
    Read backup-wide metadata from a header at byte offset `base`. Works for the
    modern 'Qb' header (encrypted tail) and the antique plaintext 'QB' header.
    For the modern format `base` is normally 0 (disk 1) but may point at the
    'Qbc2' backup-catalog header on a last disk, which is also a full header.
    The embedded name/version fields only exist on a full first-cylinder header
    (disk 1 or the backup catalog); elsewhere the tail is file data, so we only
    trust those fields when the header looks like a real one. `seed_override`
    lets a recovered (brute-forced) seed be used instead of the on-disk byte.
    """
    info = {'format': fmt,
            'id': bytes(full_file[base:base + 4]).decode('latin-1', 'replace')}
    info['disk_num'] = int(full_file[base + 4]) if len(full_file) > base + 4 else 0
    info['date'] = days_to_date(
        int.from_bytes(bytes(full_file[base + 6:base + 10]), 'big'))

    if fmt == 'antique':
        info.update(version='V4.x (antique "QB")', encrypted=False,
                    name='', vol='', num_vols=None, password=False)
        return info

    seed = int(full_file[base + 0xD]) if seed_override is None else int(seed_override)
    info.update(seed=seed, encrypted=True, num_vols=None, compress_level=0,
                password=False, name='', vol='', version='modern "Qb"')
    info['cat_size'] = int.from_bytes(bytes(full_file[base + 20:base + 24]), 'big')

    end = base + 220
    hdr = bytes(decrypt_data(full_file[base + 24:end], seed)) \
        if len(full_file) >= end else b''
    # A full header here means this is disk 1 or the backup catalog. Detect that
    # by a plausible version byte (0-2) before trusting name/version fields.
    if len(hdr) >= 196 and hdr[0] <= 2:
        ver = hdr[0]
        info['version'] = {0: '5.0/5.0.1', 1: '5.0.2',
                           2: '5.0.2'}.get(ver, f'v{ver}') + ' (modern "Qb")'
        info['num_vols'] = struct.unpack_from('>H', hdr, 2)[0]
        info['compress_level'] = hdr[4]
        info['password'] = any(hdr[5:16])
        info['name'] = _printable(
            hdr[116:156].split(b'\x00', 1)[0].decode('latin-1', 'replace'))
        info['vol'] = _printable(
            hdr[156:196].split(b'\x00', 1)[0].decode('latin-1', 'replace'))
    return info


def summarize_markers(file_list):
    """Tally data markers by tag, note LZW widths, and count decompression
    failures (compressed files whose output didn't reach the expected size)."""
    by_tag = {'FMRK': 0, 'IMRK': 0, 'CFM': 0, 'CIM': 0}
    widths = set()
    decomp_failed = 0
    for m in file_list:
        by_tag[m[4]] = by_tag.get(m[4], 0) + 1
        if m[4] in COMPRESSED_TAGS and m[7]:
            widths.add(m[7])
        if str(m[5]).startswith("Bad size"):
            decomp_failed += 1
    return {'total': len(file_list), 'by_tag': by_tag,
            'widths': sorted(widths), 'decomp_failed': decomp_failed}


def render_report(version, set_meta, recomb_meta, hdr, marker_meta, catalog_desc):
    """Print one consolidated, technical metadata block for the run."""
    line = "=" * 64
    out = [line, f" AmigaQB_extract v{version}  -  Quarterback backup analysis", line]

    def row(label, value):
        out.append(f" {label:<17}: {value}")

    size = set_meta.get('combined_size', 0)
    row("Input", f"{set_meta['num_files']} file(s), {size:,} bytes combined")

    if recomb_meta and recomb_meta.get('multidisk'):
        nums = [str(d[1]) for d in set_meta['disks'] if d[1]]
        row("Disk set", f"multi-disk - {recomb_meta['disks_reassembled']} disks "
                         f"[{', '.join('#' + n for n in nums)}], reassembled")
    elif set_meta['num_files'] > 1:
        row("Disk set", f"{set_meta['num_files']} inputs ({set_meta['order_basis']})")
    else:
        d = set_meta['disks'][0] if set_meta['disks'] else (None, None, 0)
        suffix = f" (disk #{d[1]})" if d[1] else ""
        row("Disk set", f"single disk{suffix}")

    row("Format", f"Quarterback {hdr['version']}")
    if hdr.get('name'):
        row("Backup name", f'"{hdr["name"]}"')
    if hdr.get('vol'):
        nv = hdr.get('num_vols')
        row("Source volume", f'"{hdr["vol"]}"' + (f"  ({nv} volume(s))" if nv else ""))
    row("Backup date", hdr['date'])

    if marker_meta['by_tag']['CFM'] or marker_meta['by_tag']['CIM']:
        w = marker_meta['widths']
        row("Compression", f"LZW {'/'.join(str(x) for x in w)}-bit"
                           if w else "LZW (compressed markers present)")
    elif hdr.get('compress_level'):
        row("Compression", f"LZW {hdr['compress_level'] & 0x3f}-bit (per header)")
    else:
        row("Compression", "off")

    if not hdr['encrypted']:
        row("Encryption", "none (plaintext catalog)")
    else:
        pw = ", password-protected" if hdr.get('password') else ", no password"
        row("Encryption", f"catalog encrypted (seed {hdr['seed']:#04x}){pw}")

    row("Catalog", catalog_desc)

    if marker_meta['total']:
        bt = marker_meta['by_tag']
        row("Data markers", f"{marker_meta['total']} total  "
                            f"(FMRK {bt['FMRK']} - CFM {bt['CFM']} - "
                            f"IMRK {bt['IMRK']} - CIM {bt['CIM']})")
    else:
        row("Data markers", "none found")

    warnings = list(set_meta.get('warnings', []))
    if recomb_meta:
        warnings += recomb_meta.get('warnings', [])
    if marker_meta.get('decomp_failed'):
        warnings.append(f"{marker_meta['decomp_failed']} compressed file(s) "
                        "did not decompress to full size (corruption); the "
                        "valid leading portion is still written")
    for w in warnings:
        out.append(f" ! Warning        : {w}")

    out.append(line)
    text = "\n".join(out)
    try:
        print(text)
    except UnicodeEncodeError:
        # Fall back for limited console encodings (e.g. Windows cp1252).
        print(text.encode('ascii', 'replace').decode('ascii'))
    return text


def detect_fill_runs(full_file, min_len=4096):
    """
    Find long runs of a single repeated byte, which on a 35-year-old disk image
    usually mark sectors that couldn't be read and were filled by the imaging
    tool (commonly 0x00). These are reported as suspect regions so files that
    overlap them can be flagged - it's a heuristic (real data can have long
    runs too), so nothing is discarded, only noted. Returns [(start, end), ...].
    """
    n = len(full_file)
    if n == 0:
        return []
    runs = []
    i = 0
    arr = full_file
    while i < n:
        j = i + 1
        while j < n and arr[j] == arr[i]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def overlaps_any(start, end, regions):
    """True if [start, end) intersects any (s, e) region (regions sorted)."""
    for s, e in regions:
        if start < e and s < end:
            return True
        if s >= end:
            break
    return False


def write_recovery_report(report_text, records, summary, path):
    """
    Write a per-file recovery manifest (the analysis block + a status line for
    every recovered/suspect file + a summary) to `path`. Keeps the console
    quiet but gives technical users a complete, auditable record.
    """
    lines = [report_text, "", "Per-file recovery status", "-" * 64]
    for r in records:
        lines.append(f" [{r['status']:<14}] {r.get('size', ''):>9}  {r['path']}")
    lines.append("-" * 64)
    lines.append(summary)
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"Could not write recovery report {path}: {e}")


def main():
    """
    Main function for processing Amiga QB files.
    This function reads a backup file and extracts information about files present in the backup.
    It searches for QB IDs and file headers in the backup file and extracts relevant
    information such as file names, file sizes, and file types. It also performs decompression
    of compressed QB files.
    """

    if sys.version_info < REQUIRED_PYTHON:
        sys.stderr.write(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} or higher is required.\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Restore files from an Amiga Quarterback backup file.")
    parser.add_argument(
        "backup_file", nargs='+',
        help=("One or more backup inputs to process. Pass several disk .adf "
              "files, or a directory containing them, and they are combined "
              "automatically in disk order - no need to 'cat' them together "
              "first. A single already-combined file also works."))

    parser.add_argument(
        "--ignore-catalog", "--no-catalog",
        dest="ignore_catalog",
        action="store_true",
        help=(
            "Ignore the catalog entirely and recover every file directly from "
            "its data marker, saved by filename (duplicate names kept unique so "
            "nothing is overwritten). By default no flag is needed: the catalog "
            "is found and used automatically - the primary catalog on the first "
            "disk, falling back to the backup catalog on the last disk, then to "
            "marker-only recovery - so a catalog is never required to get the "
            "data out. Use this only when you want a flat dump and don't trust "
            "the catalogued paths."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Script Version: {__version__}")

    args = parser.parse_args()

    # The output folder (DEFAULT_PATH) is created lazily by the writers, so an
    # input with nothing to recover leaves no empty qb_dump behind.

    # Load and combine all inputs.
    full_file, set_meta = load_inputs(args.backup_file)
    logging.debug("Combined image size is %s bytes", len(full_file))

    # The antique V4.x 'QB' format is structured completely differently
    # (plaintext catalog, no markers) and uses its own extraction path.
    if detect_format(full_file) == 'antique':
        extract_antique(full_file, args, set_meta)
        return

    full_file, recomb_meta = detect_multidisk(full_file)
    hdr = parse_backup_header(full_file, 'modern')
    log_phase("Scanning for data markers...")
    offset_list = find_markers(full_file)
    file_list = uncompress_data(full_file, extract_file_info(full_file, offset_list))
    marker_meta = summarize_markers(file_list)
    extra_warnings = []
    fuzzy_n = sum(1 for e in offset_list if e.get('fuzzy'))
    if fuzzy_n:
        extra_warnings.append(
            f"{fuzzy_n} marker(s) recovered via fuzzy tag match (corrupt tag byte)")
    suspect_regions = detect_fill_runs(full_file)
    suspect_files = sum(1 for m in file_list
                        if overlaps_any(m[0], m[0] + 40 + m[2], suspect_regions))
    if suspect_files:
        extra_warnings.append(
            f"{suspect_files} file(s) overlap long uniform-byte runs (possible "
            "unreadable/filled sectors); flagged 'suspect' in the recovery report")
    first_marker = offset_list[0]['offset'] if offset_list else len(full_file)

    # Resolve which catalog (if any) drives extraction - primary, then the
    # backup on the last disk, then markers-only - parsing it without writing
    # files yet, so the report can show the catalog before extraction begins.
    dir_fibs, hdr, catalog_desc, cat_warnings = resolve_catalog(
        full_file, first_marker, args.ignore_catalog)
    extra_warnings += cat_warnings

    if extra_warnings:
        set_meta = dict(set_meta)
        set_meta['warnings'] = list(set_meta.get('warnings', [])) + extra_warnings
    report_text = render_report(__version__, set_meta, recomb_meta, hdr,
                                marker_meta, catalog_desc)

    # Nothing recoverable: no usable catalog AND no data markers. Say so plainly
    # and leave no empty output folder behind. This is reached quickly because
    # all the scanning above is bounded - it never hangs on unidentifiable input.
    if not dir_fibs and not file_list:
        print("\nNothing to extract: no readable catalog and no data markers "
              "were found. If this is a Quarterback backup, the disk(s) may be "
              "too damaged, or this may not be the first/last disk of the set.")
        return

    # Write the files, collecting a per-file record for the recovery report.
    # resolve_catalog chose dir_fibs (catalog-matched) or None (markers-only);
    # exactly one writer runs.
    if dir_fibs:
        log_phase("Writing files (catalog-matched)...")
        records = match_and_save_files(dir_fibs, file_list,
                                       suspect_regions=suspect_regions)
        set_directory_timestamps(dir_fibs)
    else:
        records = process_file_markers(file_list, default_path=DEFAULT_PATH,
                                       suspect_regions=suspect_regions)

    counts = {}
    for r in records:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    summary = "Recovered " + str(len(records)) + " files: " + (
        ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none")
    write_recovery_report(report_text, records, summary,
                          str(Path(DEFAULT_PATH) / "_recovery_report.txt"))


if __name__ == "__main__":
    main()
