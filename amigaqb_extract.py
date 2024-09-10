"""
This script takes a Central Coast Software Quarterback backup disk, or a concatenated set of backup
disks, as input, and identifies which files are stored, and decompresses them into a qb_dump folder

Contact author Keith Monahan keith@techtravels.org with constructive feedback.
Bug reports should be filed as github issues. Please include the problem ADF backup file.
https://github.com/keithgh1/AmigaQB_extract

Copyright (c) 2024 Keith Monahan
Licensed under the MIT License. See LICENSE file in the project root for full license information.
"""
import argparse
import sys
import os
import logging
import struct
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathvalidate import sanitize_filename, sanitize_filepath

# logging.basicConfig(FILENAME='qb_event.log', encoding='utf-8', level=logging.DEBUG)
__version__ = "0.4.2"

# Minimum required version
REQUIRED_PYTHON = (3, 6)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

# set flags for the DirFib structure, but used outside the class, so necessary here
FLAG_DIR_MASK = 128
FLAG_SEL_MASK = 64
FLAG_ERR_MASK = 32
FLAG_ODD_MASK = 16    # Obsolete, reserved
FLAG_PART_MASK = 8
FLAG_HLINK_MASK = 4
FLAG_SLINK_MASK = 2
FLAG_BITS_MASK = 1    # Backup-Compress or not, Restore-read-protected?


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

        # Extract null-terminated strings df_name and df_comment
        df_name = DirFib._extract_null_terminated_string(byte_list, offset)
        offset += len(df_name) + 1

        df_comment = DirFib._extract_null_terminated_string(byte_list, offset)
        offset += len(df_comment) + 1

        # Create a DirFib instance
        return DirFib(df_size1, df_size2, *remaining_fields, df_name, df_comment), offset

    @staticmethod
    def _extract_null_terminated_string(byte_list, offset):
        """
        Extracts a null-terminated string from the byte list, retaining extended characters
        and sanitizing only invalid ones.
        """
        end = offset
        while end < len(byte_list) and byte_list[end] != 0x00:
            end += 1
        if end >= len(byte_list):
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


def parse_dir_fibs(byte_list, header_length='20'):
    """
    Parse the directory FIBs from a byte list.
    """
    offset = 222  # Starting offset; adjust as needed

    dir_fibs = []
    while offset < len(byte_list):
        dir_fib, new_offset = DirFib.from_bytes(
            byte_list, offset, header_length)
        dir_fibs.append(dir_fib)
        offset = new_offset  # Update offset to the next structure

    return dir_fibs


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
    """Decrypt a list of byte values."""
    return [decrypt_byte(byte, encrypt_val) for byte in data]


def generate_path(path_stack, filename, os_type='linux'):
    """ Define the separator based on the operating system type """
    separator = '\\' if os_type.lower() == 'windows' else '/'

    # Join the paths from the path stack using the correct separator
    # No leading separator, and separators between all path components
    path = separator.join([p[0] for p in path_stack]) + separator + filename
    return path


def process_dirfibs(dirfibs):
    """
    Process a list of directory file information blocks.
    """
    path_stack = [(DEFAULT_PATH, 0xBADCAFE)]

    for dir_fib in dirfibs:
        if path_stack[-1][1] != 0xBADCAFE:
            # Create a new tuple with the updated count and replace the old one
            # to be clear, this subtracts 1 from the count for every file or directory
            # except in the case of the first entry, which is the root directory
            # The path stack tuple is defined as (directory name, remaining file count)
            # Remaining file count is initially set to df_filcnt from the
            # DirFib structure

            path_stack[-1] = (path_stack[-1][0], path_stack[-1][1] - 1)

        if dir_fib.df_flags & FLAG_DIR_MASK:  # Directory
            path_stack.append((dir_fib.df_name, dir_fib.df_filcnt))
            dir_path = generate_path(path_stack, '', os_type='windows')

            # sanitizing the dir_path here helps with two things:
            # 1) making sure that the path is valid, so the os.makedirs() is less likely to fail
            # 2) making sure that our dirfib structure never has invalid characters in it
            clean_path = sanitize_filepath(dir_path)
            dir_fib.df_name = clean_path  # Update the name to the full path

            try:
                os.makedirs(clean_path, exist_ok=True)
            except OSError as e:
                print(
                    f"Resetting path stack because there was an Error creating directory: {clean_path}")
                print(e)
                print("Path stack at this point is ", path_stack)
                path_stack = [(DEFAULT_PATH, 0xBADCAFE)]

        elif dir_fib.df_flags & FLAG_SEL_MASK:  # File
            file_path = generate_path(
                path_stack, dir_fib.df_name, os_type='windows')
            # Update the name to the full sanitized path including the filename
            dir_fib.df_name = sanitize_filepath(file_path)

        else:
            # this resets the path stack if we see a corrupted or unsupported flags value
            # fwiw there are other valid flags like links that are not
            # supported here
            print(
                "Resetting to root --- Unsupported flags value:",
                dir_fib.df_flags)
            print("Path stack at this point is ", path_stack)
            print("Associated filename was ", dir_fib.df_name)
            path_stack = [(DEFAULT_PATH, 0xBADCAFE)]

        while path_stack and path_stack[-1][1] == 0:
            path_stack.pop()

    return dirfibs


def match_and_save_files(dir_fibs, file_list, default_path=DEFAULT_PATH):
    """
    Iterates through DirFib entries that are files, attempts to match each entry
    with a file marker based on filename, size, and approximate position,
    and saves the matched files. Unmatched file markers are also saved separately.

    Args:
        dir_fibs (list): List of DirFib entries.
        file_list (list): List of file markers with filename, size, and content.
        default_path (str): Default directory to save files when no matching marker is found.
    """
    try:
        os.makedirs(default_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {default_path}")
        print(e)

    matched_markers = set()  # Keep track of matched markers
    file_index = 0  # Track the position in file_list

    # Iterate only through DirFib entries that are files
    for _, dir_fib in enumerate(dir_fibs):  # Using _ to ignore the index value

        if dir_fib.df_flags & FLAG_DIR_MASK:  # Skip directories
            continue

        # Check within a small range around the current position for possible
        # matches
        for offset in range(-2,
                            3):  # Allow some leeway to account for corruption shifts
            marker_index = file_index + offset
            if 0 <= marker_index < len(file_list):
                marker = file_list[marker_index]

                # Check filename and size match criteria
                if dir_fib.df_name.endswith(
                        marker[1]) and dir_fib.df_size1 == marker[2]:
                    file_path = dir_fib.df_name
                    # Assuming index 6 contains the file content
                    content = marker[6]

                    try:
                        # Ensure the directory structure exists
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        # Save the file content
                        with open(file_path, 'wb') as file:
                            file.write(content)

                        print(f"Saved file: {file_path}")

                    except OSError as e:
                        print(f"Error saving file: {file_path}")
                        print(e)

                    # Mark this marker as matched
                    # Would we want to mark this even if an error occurs?
                    # reinvestigate this in the future
                    matched_markers.add(marker_index)
                    break
        else:
            # If no match is found, save the file in the default directory as a
            # fallback
            print(
                f"No associated file marker found for catalog entry: {
                    dir_fib.df_name}")

        file_index += 1

    # Save any unmatched file markers that were not associated with a DirFib
    # entry
    for j, marker in enumerate(file_list):
        if j not in matched_markers:
            # Save the unmatched marker in the default directory
            fallback_filename = os.path.join(default_path, marker[1])
            clean_fallback_filename = sanitize_filepath(fallback_filename)
            try:
                with open(clean_fallback_filename, 'wb') as file:
                    # Save the content directly from the file marker
                    file.write(marker[6])
                    print(
                        f"Saved unmatched file marker to: {clean_fallback_filename}")
            except OSError as e:
                print(f"Error saving file: {clean_fallback_filename}")
                print(e)


def process_file_markers(file_list, default_path=DEFAULT_PATH):
    """
    Processes the file markers, ignoring the catalog and saving all files
    to the default directory.

    Args:
        file_list (list): List of file markers containing filename, size, and content.
        default_path (str): Directory where all files will be dumped.
    """
    try:
        os.makedirs(default_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {default_path}")
        print(e)

    for marker in file_list:
        # Extract filename and content from the marker
        filename = marker[1]  # contains the filename
        content = marker[6]   # contains the file content

        # Define the full path where the file will be saved in the default
        # directory
        file_path = os.path.join(default_path, filename)

        try:
            # Save the file content
            with open(file_path, 'wb') as file:
                file.write(content)

            print(f"Saved file: {file_path}")
        except OSError as e:
            print(f"Error saving file: {file_path}")
            print(e)


def hex_display(data):
    """
    Handles conversion of data to a hex string like this 0E FF C4...
    """
    output_data_string = ""

    for my_data in data:
        if len(hex(ord(my_data))) < 4:
            output_data_string += "0"

        output_data_string += hex(ord(my_data))[2:]
        output_data_string += " "

    return output_data_string


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


def uncompress_me(buf):
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
    my_string = ""

    code_size = 9

    # 2^9 - 1
    maximum_table_size = 511

    # Building and initializing the dictionary
    # ASCII 0-255 in positions 0-255

    dictionary_size = 256
    dictionary = dict([(x, chr(x)) for x in range(dictionary_size)])

    # LZW Decompression algorithm

    i = 0

    while (i < (len(bits) - 9)):

        # Don't allow code_size to grow larger than 16-bits.
        if next_code > maximum_table_size and code_size != 16:

            # need to bump the bit_pos by 9, essentially skipping a 9-bit word
            # Also bump code size by 1, and recalculate the maximum code
            # Only skip the 9-10 transition? not others? OK!
            if code_size == 9:
                i += code_size

            code_size += 1
            maximum_table_size = pow(2, int(code_size)) - 1

            logging.debug("Code size change at bits: %s", i)
            logging.debug("code_size is %s", code_size)
            logging.debug(
                "Approximate decompress size is %s",
                len(decompressed_data))

        code = get_code(buf, i, code_size)

        if code == 0xDEADBEEF:
            return str.encode("DEADBEEF")

        if code > next_code:
            logging.debug(
                "Code read is higher than it should be. code is %s next_code is %s",
                code,
                next_code)

        if code not in dictionary:
            dictionary[code] = my_string + (my_string[0])

        # This generates a ton of log entries
        # logging.debug("Decomp-pos:"+str(len(decompressed_data)))
        # logging.debug("out:"+hex_display(dictionary[code]))

        decompressed_data += dictionary[code]

        if len(my_string) != 0:
            dictionary[next_code] = my_string + (dictionary[code][0])
            next_code += 1
            logging.debug("next_code is now %s", next_code)

        my_string = dictionary[code]

        i += code_size

    output_data_string = ""

    # Shouldn't we be using hex_display() for this?
    for data in decompressed_data:
        if len(hex(ord(data))) < 4:
            output_data_string += "0"

        output_data_string += hex(ord(data))[2:]
        output_data_string += " "

    return bytearray.fromhex(output_data_string)


def initialize_dataframe():
    """Initializes the pandas DataFrame to store file information."""
    d = {
        'Offset': [],
        'FILENAME': [],
        'Filesize1': [],
        "Filesize2": [],
        "Filecontent": []}
    return pd.DataFrame(data=d)


def load_file(filepath):
    """Loads the binary file into a numpy array."""
    return np.fromfile(filepath, dtype=np.ubyte)


def detect_multidisk(full_file):
    """Detects if the file is part of a multidisk set and removes QB headers."""
    qb_list = []

    if len(full_file) < 901121:
        print("Single disk detected.")
    else:
        print("Multidisk detected, looking for QB ids...")

        search_val = [0x51, 0x62]
        n = len(search_val)
        possibles = np.where(full_file == search_val[0])[0]

        for p in possibles:
            check = full_file[p:p + n]
            if np.all(check == search_val):
                if not p % 901120:
                    qb_list.append((p, p + 16))

        print("QB id's found:", len(qb_list))
        logging.debug("QB ids found: %s", len(qb_list))
        logging.debug(qb_list)
        print(qb_list)

        if len(qb_list) > 0:
            full_file = np.delete(full_file, np.concatenate(
                [np.arange(start, end) for start, end in qb_list]))

    return full_file


def find_markers(full_file):
    """Finds CFM90 and FMRK markers in the binary file."""
    search_val1 = {'value': [0x43, 0x46, 0x4D, 0x90], 'tag': 'CFM90'}
    search_val2 = {'value': [0x46, 0x4D, 0x52, 0x4B], 'tag': 'FMRK'}

    search_values = [search_val1, search_val2]

    offset_list = []

    for search_val in search_values:
        n = len(search_val['value'])
        possibles = np.where(full_file == search_val['value'][0])[0]

        for p in possibles:
            check = full_file[p:p + n]
            if np.all(check == search_val['value']):
                offset_list.append({'tag': search_val['tag'], 'offset': p})

    logging.debug(offset_list)
    return offset_list


def extract_file_info(full_file, offset_list):
    """Extracts file information based on found markers."""
    file_list = []

    for offset in offset_list:
        tag = offset['tag']
        offset = offset['offset']

        filename = "".join(
            [chr(item) for item in full_file[offset + 4:offset + 34]]).split('\x00', 1)[0]

        filesize1_bits = np.unpackbits(full_file[offset + 36:offset + 40])
        filesize1 = filesize1_bits.dot(
            2 ** np.arange(filesize1_bits.size)[::-1])

        file_list.append([offset, filename, filesize1, 0, tag, '', ''])

    return file_list


def uncompress_data(full_file, file_list):
    """Uncompresses files marked with CFM90 and also extracts data from uncompressed files."""
    for j in range(len(file_list)):
        if file_list[j][4] == 'CFM90':
            cfm_offset = file_list[j][0]
            end_lzw_datastream = determine_end_of_datastream(
                full_file, file_list, j)

            start_lzw_datastream = cfm_offset + 40

            file_list[j][6] = uncompress_me(
                full_file[start_lzw_datastream:end_lzw_datastream])

            if len(file_list[j][6]) == file_list[j][2]:
                # print("Size check good.\n")
                file_list[j][5] = "Decompressed OK"
            else:
                file_list[j][6] = uncompress_me(
                    full_file[start_lzw_datastream:end_lzw_datastream + 1])
                if len(file_list[j][6]) == file_list[j][2]:
                    # print("Size check good.\n")
                    file_list[j][5] = "Decompressed OK on +1 retry"
                else:
                    print("ERROR Size expected:", file_list[j][2])
                    print("ERROR Size returned:", len(file_list[j][6]))
                    if file_list[j][6] == "DEADBEEF":
                        print("Bad code_size detected")
                    file_list[j][5] = "Bad Size"

        elif file_list[j][4] == 'FMRK':
            file_list[j][6] = full_file[file_list[j][0] +
                                        40:file_list[j][0] + file_list[j][2] + 40]

    return file_list


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
    parser.add_argument("backup_file", help="The backup filename to process")

    parser.add_argument(
        "--catalog",
        choices=['primary', 'backup', 'ignore'],
        default='primary',
        help=(
            "Choose how to handle the catalog: "
            "'primary' (default) will process the primary catalog at the beginning of the backup file, "
            "'backup' will process the backup catalog, "
            "'ignore' will ignore the catalog and dump all restored files into the root qb_dump directory."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Script Version: {__version__}")

    parser.add_argument(
        "--header-length",
        choices=['16', '20'],
        default='20',
        help=(
            "Specify the header length of DirFib entries: "
            "'16' bytes for entries with one file size, "
            "'20' bytes (default) for entries with two file sizes."
        )
    )

    args = parser.parse_args()

    # we need to do this no matter what option is chosen

    try:
        full_file = load_file(args.backup_file)
        logging.debug(
            "Opened file: %s File size is %s",
            sys.argv[1],
            len(full_file))
    except OSError as e:
        print(f"Error opening backup file: {args.backup_file}")
        print(e)

    full_file = detect_multidisk(full_file)
    offset_list = find_markers(full_file)
    file_list = extract_file_info(full_file, offset_list)

    file_list = uncompress_data(full_file, file_list)

    print("We found ", len(file_list), " file markers in the backup file.")

    print(pd.DataFrame(file_list, columns=[
        'Offset', 'FILENAME', 'Filesize1', 'Filesize2', 'FileType', 'CompressedData', 'UncompressedData']
    )[['Offset', 'FILENAME', 'Filesize1', 'FileType']])

    if args.catalog == 'ignore':
        print("Ignoring the file catalog and processing only file markers.")

        process_file_markers(file_list, default_path=DEFAULT_PATH)

    elif args.catalog == 'backup':
        print("Processing the backup catalog. This feature is unsupported for now.")
        # Add your logic to process the backup catalog
    else:
        print("Processing the primary catalog at the beginning of the backup file.")

        # first we must decrypt the catalog, which would be up to the first marker found
        # 0xD is the location of the encryption value in the QB header
        decrypted_catalog = decrypt_data(
            full_file[0:offset_list[0]['offset']], full_file[0xD])

        # with open("decryptedcat.bin", "wb") as binary_file:
        # binary_file.write(bytes(decrypted_catalog))

        # Parse DirFib entries using the specified header length
        dir_fibs = parse_dir_fibs(
            decrypted_catalog, header_length=args.header_length)

        # Determine directory structure and create directories
        dir_fibs = process_dirfibs(dir_fibs)

        # Print details of each DirFib structure
        """
        for i, dir_fib in enumerate(dir_fibs):

            print(f"DirFib {i + 1}:")
            print("  df_size1:", str(dir_fib.df_size1))
            print("  df_size2:", str(dir_fib.df_size2))
            print("  df_days:", dir_fib.df_days)
            print("  Calculated Date:", dir_fib.date.strftime('%Y-%m-%d'))
            print("  df_minutes:", dir_fib.df_minutes)
            print("  df_ticks:", dir_fib.df_ticks)
            print("  df_filcnt:", dir_fib.df_filcnt)
            print("  df_prot:", dir_fib.df_prot)
            print(
            f"  df_flags: {
                dir_fib.df_flags} ({
                dir_fib.flags_to_string()})")
            print("  df_name:", dir_fib.df_name)
            print("  df_comment:", dir_fib.df_comment)
            print()
        """

        file_count = sum(bool(not (f.df_flags & FLAG_DIR_MASK))
                         for f in dir_fibs)
        print("We found ", file_count, " file entries in the catalog.")
        match_and_save_files(dir_fibs, file_list)


if __name__ == "__main__":
    main()
