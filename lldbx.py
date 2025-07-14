################################################################################
# LLDB eXtended
# Improve the LLDB interface without any dependency.

import lldb
import struct
import re
import json
from json.decoder import JSONDecodeError
import os

################################################################################
# Configuration

REGISTER_COMMANDS = True  # Don't need to precede 'lldbx' before the commands

GLOBAL_CFGS_DEFAULT = {  # Dynamic configuration
    "color": True,
    "color_dict": {
        "default": "\033[0m",
        "separator": "\033[34m",
        "poffsets_name": "\033[35m",
        "poffsets_type": "\033[32m",
        "poffsets_padding": "\033[31m",
        "telescope_addr": "\033[34m",
        "reg_updated": "\033[30m",
        "reg_used": "\033[32m",
        "code_prev": "\033[30m",
        "code_current": "\033[32m",
        "hex_zero": "\033[30m",
        "hex_printable": "\033[33m",
        "hex_other": "\033[36m",
        "analyze_branch": "\033[35m",
    },
    "256colors": True,
}

CONFIG_ENCODING = "utf-8"
CONFIG_DIR = ".lldbx"
CONFIG_FILENAME = "config.json"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), CONFIG_DIR)


def configuration_merge(loaded_conf):
    update = False

    # merging new keys only
    for key in GLOBAL_CFGS_DEFAULT.keys():
        if key not in loaded_conf:
            loaded_conf[key] = GLOBAL_CFGS_DEFAULT[key]
            update = True

    return update


def configuration_drop(conf_out):
    if not os.path.exists(CONFIG_PATH):
        os.mkdir(CONFIG_PATH)

    drop_file = os.path.join(CONFIG_PATH, CONFIG_FILENAME)

    with open(drop_file, "wb") as fd_out:
        fd_out.write(
            json.dumps(conf_out, sort_keys=True, indent=4).encode(CONFIG_ENCODING)
        )

    print("(lldbx) Configuration file dropped as: %s" % drop_file)


def configuration_load():
    print("(lldbx) Configuration load")

    conf_file = os.path.join(CONFIG_PATH, CONFIG_FILENAME)

    if os.path.exists(conf_file):
        with open(conf_file) as fd_in:
            try:
                loaded_conf = json.load(fd_in)
                return loaded_conf
            except JSONDecodeError as e:
                print(
                    "(lldbx) Failed to load configuration (JSONDecodeError), defaulting to global configuration"
                )
                print("(lldbx) Please review or delete file: %s" % conf_file)

    else:
        print("(lldbx) Configuration file does not exist, dropping a fresh one")
        configuration_drop(GLOBAL_CFGS_DEFAULT)

    return GLOBAL_CFGS_DEFAULT


GLOBAL_CFGS = configuration_load()

################################################################################
# Color management


def color(name="default"):
    if GLOBAL_CFGS["color"] and (name in GLOBAL_CFGS["color_dict"]):
        return GLOBAL_CFGS["color_dict"][name]

    return ""


def fnv1a(s):
    # Constants for FNV-1a 64-bit hash
    FNV_prime = 0x100000001B3
    offset_basis = 0xCBF29CE484222325

    hash = offset_basis
    for x in s.encode("utf-8"):
        hash ^= x
        hash *= FNV_prime
        hash &= 0xFFFFFFFFFFFFFFFF  # Ensure 64-bit hash value

    return hash


def fnv1a_with_modulo(s, mod):
    hash_value = fnv1a(s)
    hash_value = hash_value ^ (hash_value >> 32)
    return hash_value % mod


def string_to_ansi_color(string):
    """Always return the same color for a specific string"""
    if not GLOBAL_CFGS["color"]:
        return string

    if GLOBAL_CFGS["256colors"]:
        color_code = fnv1a_with_modulo(string, 256)
        return f"\033[38;5;{color_code}m{string}\033[0m"

    color_code = fnv1a_with_modulo(string, 8) + 30
    return f"\033[{color_code}m{string}\033[0m"


def colorize_substrings(string, substrings):
    # Sort the substrings by length, longest first, to prevent partial replacements
    substrings.sort(key=len, reverse=True)

    def replacer(match):
        substring = match.group(0)
        return string_to_ansi_color(substring)

    # Build a pattern that matches any of the substrings
    pattern = "|".join(re.escape(sub) for sub in substrings)
    return re.sub(pattern, replacer, string)


def colorize_arm64e_regs(string):
    if not GLOBAL_CFGS["color"]:
        return string

    regs = [
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "fp",
        "lr",
        "sp",
        "pc",
        "cpsr",
        "w0",
        "w1",
        "w2",
        "w3",
        "w4",
        "w5",
        "w6",
        "w7",
        "w8",
        "w9",
        "w10",
        "w11",
        "w12",
        "w13",
        "w14",
        "w15",
        "w16",
        "w17",
        "w18",
        "w19",
        "w20",
        "w21",
        "w22",
        "w23",
        "w24",
        "w25",
        "w26",
        "w27",
        "w28",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "fpsr",
        "fpcr",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s16",
        "s17",
        "s18",
        "s19",
        "s20",
        "s21",
        "s22",
        "s23",
        "s24",
        "s25",
        "s26",
        "s27",
        "s28",
        "s29",
        "s30",
        "s31",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "d9",
        "d10",
        "d11",
        "d12",
        "d13",
        "d14",
        "d15",
        "d16",
        "d17",
        "d18",
        "d19",
        "d20",
        "d21",
        "d22",
        "d23",
        "d24",
        "d25",
        "d26",
        "d27",
        "d28",
        "d29",
        "d30",
        "d31",
        "far",
        "esr",
        "exception",
        "amx_x0",
        "amx_x1",
        "amx_x2",
        "amx_x3",
        "amx_x4",
        "amx_x5",
        "amx_x6",
        "amx_x7",
        "amx_y0",
        "amx_y1",
        "amx_y2",
        "amx_y3",
        "amx_y4",
        "amx_y5",
        "amx_y6",
        "amx_y7",
        "amx_z0",
        "amx_z1",
        "amx_z2",
        "amx_z3",
        "amx_z4",
        "amx_z5",
        "amx_z6",
        "amx_z7",
        "amx_z8",
        "amx_z9",
        "amx_z10",
        "amx_z11",
        "amx_z12",
        "amx_z13",
        "amx_z14",
        "amx_z15",
        "amx_z16",
        "amx_z17",
        "amx_z18",
        "amx_z19",
        "amx_z20",
        "amx_z21",
        "amx_z22",
        "amx_z23",
        "amx_z24",
        "amx_z25",
        "amx_z26",
        "amx_z27",
        "amx_z28",
        "amx_z29",
        "amx_z30",
        "amx_z31",
        "amx_z32",
        "amx_z33",
        "amx_z34",
        "amx_z35",
        "amx_z36",
        "amx_z37",
        "amx_z38",
        "amx_z39",
        "amx_z40",
        "amx_z41",
        "amx_z42",
        "amx_z43",
        "amx_z44",
        "amx_z45",
        "amx_z46",
        "amx_z47",
        "amx_z48",
        "amx_z49",
        "amx_z50",
        "amx_z51",
        "amx_z52",
        "amx_z53",
        "amx_z54",
        "amx_z55",
        "amx_z56",
        "amx_z57",
        "amx_z58",
        "amx_z59",
        "amx_z60",
        "amx_z61",
        "amx_z62",
        "amx_z63",
        "amx_state",
    ]
    regs += ["x29", "x30", "x31"]  # They are used in the disassembly too
    regs.sort(key=len, reverse=True)

    # Don't take the one numbers like 0x1 as register name
    for i in range(len(regs)):
        regs[i] = f"\\b{regs[i]}\\b"

    def replacer(match):
        reg = match.group(0)
        original_reg = reg

        if len(reg) > 1 and reg[0] == "w" and reg[1].isdigit():
            # We want w9 and x9 to be the same color
            reg = "x" + reg[1:]
        if len(reg) > 1 and reg[0] == "s" and reg[1].isdigit():
            reg = "d" + reg[1:]
        elif reg == "wzr":
            reg = "xzr"

        result = string_to_ansi_color(reg)

        if reg != original_reg:
            # Restore the register name but keep the color
            result = result.replace(reg, original_reg)
        return result

    # Build a pattern that matches any of the substrings
    pattern = "|".join(sub for sub in regs)
    return re.sub(pattern, replacer, string)


def colorize_arm64e_mnemonic(mnemonic):
    if not GLOBAL_CFGS["color"]:
        return mnemonic

    branchs = ["cbz", "cbnz", "tbz", "tbnz", "ret", "eret", "retab"]

    if mnemonic[0] == "b" or mnemonic in branchs:
        return color("analyze_branch") + mnemonic + color()

    return mnemonic


################################################################################
# Command analyze -- Analyze a function


def cmd_analyze(debugger, args, result, internal_dict):
    """Analyze a function and show a more readable disassembled version."""
    args = args.split()
    if len(args) < 1:
        print("Usage: analyze <addr> [min_lines=10] [max_lines=-1]")
        return

    dbg = Debugger(debugger)
    function_addr = dbg.expression(args[0]).GetValueAsUnsigned()

    # The number of minimum lines will be used if the symbol is not found
    min_lines = 4
    max_lines = -1
    if len(args) > 1:
        min_lines = int(args[1], 0)
    if len(args) > 2:
        max_lines = int(args[2], 0)

    target = dbg.target()
    resolved_addr = target.ResolveLoadAddress(function_addr)
    module = resolved_addr.GetModule()
    module_name = module.GetFileSpec().GetFilename()

    sym = resolved_addr.GetSymbol()
    instructions = sym.GetInstructions(target)
    full_symbol_name = "%s`%s:" % (module_name, sym.GetDisplayName())

    if len(instructions) == 0:
        print(f"No instructions to show at 0x{function_addr:x}")
        return

    if len(instructions) == 0:
        # This symbol has not been found
        offset = function_addr - module.GetSectionAtIndex(0).GetLoadAddress(target)
        full_symbol_name = "%s+0x%x:" % (module_name, offset)

    if len(instructions) < min_lines:
        # We want more lines that this symbol only
        instructions = target.ReadInstructions(resolved_addr, min_lines)

    inst_pos = -1
    i = 0
    for inst in instructions:
        # TODO: get closest line
        if inst.GetAddress().GetLoadAddress(target) == function_addr:
            inst_pos = i
            break
        i += 1

    first_inst_pos = 0
    last_inst_pos = len(instructions)
    if max_lines != -1:
        last_inst_pos = min(len(instructions), max_lines)

    # Add the lines and the comments to be able to align them
    lines_no_color = []
    lines = []
    comments = {}
    for i in range(len(instructions)):
        inst = instructions[i]
        mnemonic = inst.GetMnemonic(target)
        operands = inst.GetOperands(target)
        comment = dbg.get_comment(target, inst)
        addr = inst.GetAddress().GetLoadAddress(target)
        if first_inst_pos <= i <= last_inst_pos:
            if i == first_inst_pos:
                print(full_symbol_name)

            # First put the color on the address
            spaces = "   "
            if i == inst_pos:
                spaces = "-> "

            line_color = "default"
            if i < inst_pos:
                line_color = "code_prev"
            elif i == inst_pos:
                line_color = "code_current"

            line_no_color = f"0x{addr:x}"
            line = spaces + color(line_color) + f"0x{addr:x}" + color()

            # Now the mnemonic
            spaces = " " * max(8 - len(mnemonic), 0)
            line_no_color += f": {mnemonic}{spaces} "
            line += f": {colorize_arm64e_mnemonic(mnemonic)}{spaces} "

            # The operands
            line_no_color += operands
            line += colorize_arm64e_regs(operands)

            # The comment
            if len(comment) > 0:
                comments[len(lines)] = comment

            lines_no_color.append(line_no_color)
            lines.append(line)

    # Show the lines
    biggest_line_len = max([len(line) for line in lines_no_color])
    for i in range(len(lines)):
        line = lines[i]
        if i in comments:
            line += " " * (biggest_line_len - len(lines_no_color[i]))
            line += "    ; " + comments[i]
        print(line)


################################################################################
# Command break

def cmd_br(debugger, args, result, internal_dict):
    """Add a breakpoint on an address or a symbol."""
    args = args.split()
    if len(args) < 1:
        print("Usage: br <addr|symbol>")
        return

    # Could be a symbol or an expression
    symbol = args[0]
    dbg = Debugger(debugger)
    addr = dbg.expression(symbol).GetValueAsUnsigned()

    target = dbg.target()
    if not target:
        print("ERR: The target is not ready")
        return

    if addr:
        br = target.BreakpointCreateByAddress(addr)
        print("%d locations breakpoint added" % br.GetNumLocations())
        return

    module_name = None
    if "`" in symbol:
        values = symbol.split("`")
        if len(values) == 2:
            module_name = values[0]
            symbol = values[1]

    br = target.BreakpointCreateByName(symbol, module_name)
    print("%d locations breakpoint added" % br.GetNumLocations())


################################################################################
# Command hexdump - Hexadecimal dump


def cmd_hexdump(debugger, args, result, internal_dict):
    """Hexadecimal dump."""
    args = args.split()
    if len(args) < 1:
        print("Usage: hexdump <addr> [size=0x40]")
        return

    dbg = Debugger(debugger)
    base_addr = dbg.expression(args[0]).GetValueAsUnsigned()
    size = 0x40
    if len(args) > 1:
        size = int(args[1], 0)

    try:
        memory = dbg.read_memory(base_addr, size)
    except Exception as err:
        print(err)
        return

    hex_line = ""
    ascii_line = ""
    addr = base_addr
    line_width = 16

    for i in range(len(memory)):
        if memory[i] == 0:
            ascii_line += f"{color('hex_zero')}.{color()}"
            hex_line += f"{color('hex_zero')}{memory[i]:02x}{color()} "
        elif 32 < memory[i] < 127:  # Printable
            ascii_line += f"{color('hex_printable')}{memory[i]:c}{color()}"
            hex_line += f"{color('hex_printable')}{memory[i]:02x}{color()} "
        else:
            ascii_line += f"{color('hex_other')}.{color()}"
            hex_line += f"{color('hex_other')}{memory[i]:02x}{color()} "

        if not (i + 1) % line_width:
            print("%x: %s%s" % (addr, hex_line, ascii_line))
            hex_line = ""
            ascii_line = ""
            addr += line_width

    if len(ascii_line) > 0:
        # Show the last line, same but with spaces
        nb_spaces = (line_width - len(ascii_line)) * 3
        spaces = ""
        if nb_spaces > 0:
            spaces += " " * nb_spaces
        print("%x: %s%s%s" % (addr, hex_line, spaces, ascii_line))


################################################################################
# Command conf - Configure global settings


def cmd_conf(debugger, args, result, internal_dict):
    """Configure global settings"""
    args = args.split()

    if len(args) < 1:
        # Show the usage and the current configuration
        print("Usage: conf <config> [value]")

        print("Current configuration:")
        print(json.dumps(GLOBAL_CFGS, indent=2, sort_keys=True))
        return

    conf_key = args[0]
    if conf_key not in GLOBAL_CFGS:
        print("The configuration key '%s' doesn't exist" % conf_key)
        return

    if len(args) > 1:
        # We need to set a configuration value
        try:
            value = json.loads(args[1])
        except:
            print(f"{args[1]} is not a valid JSON value")
            return

        expected_type = type(GLOBAL_CFGS[conf_key])
        if type(value) != expected_type:
            print(
                f"{conf_key} is expected to be a {expected_type.__name__}, not a {type(value).__name__}"
            )
            return

        GLOBAL_CFGS[conf_key] = value

    # Print the current value, updated or not
    print(f"{conf_key}: {json.dumps(GLOBAL_CFGS[conf_key])}")
    return


################################################################################
# Command memset & memcpy - memory manipulation wrapper primitives


def cmd_memset(debugger, args, result, internal_dict):
    """memset equivalent primitive."""
    args = args.split()
    if len(args) < 3:
        print("Usage: memset <addr> <byte> <size>")
        return

    dbg = Debugger(debugger)
    base_addr = dbg.expression(args[0]).GetValueAsUnsigned()
    byte = dbg.expression(args[1]).GetValueAsUnsigned() & 0xFF
    size = dbg.expression(args[2]).GetValueAsUnsigned()

    print(f"> memset: Write 0x{size:x} times 0x{byte:x} to 0x{base_addr:x}")

    try:
        dbg.write_memory(base_addr, chr(byte) * size)
    except Exception as err:
        print(err)
        print(f"Can't write 0x{size:x} bytes at 0x{base_addr:x}")


def cmd_memcpy(debugger, args, result, internal_dict):
    """memcpy equivalent primitive."""
    args = args.split()
    if len(args) < 3:
        print("Usage: memcpy <dst> <src> <size>")
        return

    dbg = Debugger(debugger)
    dst = dbg.expression(args[0]).GetValueAsUnsigned()
    src = dbg.expression(args[1]).GetValueAsUnsigned()
    size = dbg.expression(args[2]).GetValueAsUnsigned()

    print(f"> memcpy: Copy 0x{size:x} bytes from 0x{src:x} to 0x{dst:x}")

    try:
        mem = dbg.read_memory(src, size)
    except:
        print(f"Can't read 0x{size:x} bytes at 0x{src:x}")
        return

    try:
        dbg.write_memory(dst, mem)
    except:
        print(f"Can't write 0x{size:x} bytes at 0x{dst:x}")


################################################################################
# Command nopac -- Remove PAC from a pointer


def nopac_kernel(ptr):
    test_pac_bit = 1 << 55
    if ptr & test_pac_bit:
        ptr = ptr | 0xFFFFFF8000000000
    return ptr


def nopac_user(ptr):
    if (ptr & 0xFFFF000000000000) > 0:
        ptr = ptr & 0x0000007FFFFFFFFF
    return ptr


def cmd_nopac(debugger, args, result, internal_dict):
    """Remove PAC from a pointer."""
    args = args.split()
    if len(args) < 1:
        print("Usage: nopac <ptr>")
        return

    dbg = Debugger(debugger)
    ptr = dbg.expression(args[0]).GetValueAsUnsigned()
    print("user:    0x%x" % nopac_user(ptr))
    print("kernel:  0x%x" % nopac_kernel(ptr))


################################################################################
# Command poffsets - Print structure offsets


def print_field_offset(field, base_offset, depth, next_field_offset=None):
    alignment = " " * ((depth + 1) * 4)
    offset = base_offset + field.GetOffsetInBytes()
    field_name = field.GetName()
    field_type = field.GetType()
    field_type_name = field_type.GetName()

    line = f"{alignment}+0x{offset:03x}"  # offset
    line += f" {color('poffsets_type')}{field_type_name:<10s}{color()}"  # type
    line += f" {field_name}"  # name
    print(line)

    # If this element is a structure, show the sub-elements
    nb_sub_fields = field_type.GetNumberOfFields()
    if nb_sub_fields > 0:
        for i in range(nb_sub_fields):
            sub_field = field_type.GetFieldAtIndex(i)

            next_subfield_offset = offset + field_type.GetByteSize()
            if i + 1 < nb_sub_fields:
                next_subfield_offset = (
                    offset + field_type.GetFieldAtIndex(i + 1).GetOffsetInBytes()
                )

            print_field_offset(sub_field, offset, depth + 1, next_subfield_offset)

    # Check if we have a padding after this field
    if next_field_offset:
        field_size = next_field_offset - offset
        type_size = field_type.GetByteSize()
        padding_size = field_size - type_size

        if padding_size > 0:
            padding_offset = offset + type_size
            line = f"{alignment}+0x{padding_offset:03x}"  # offset
            line += f" {color('poffsets_padding')}u8[{padding_size}]"  # type
            line += f"      padding{color()}"  # name
            print(line)


def print_sbtype_offset(sbtype, base_offset, depth):
    nb_parent_classes = sbtype.GetNumberOfDirectBaseClasses()
    nb_fields = sbtype.GetNumberOfFields()

    if depth > 0 and (nb_parent_classes + nb_fields == 0):
        # Nothing to print
        return

    # Print the structure or the class name
    alignment = " " * (depth * 4)
    print(
        f"{alignment}+0x{base_offset:03x} {color('poffsets_name')}{sbtype.GetName()}{color()} {{"
    )

    # Parent C++ classes
    for i in range(nb_parent_classes):
        base_class = sbtype.GetDirectBaseClassAtIndex(i)
        base_class_offset = base_class.GetOffsetInBytes()

        # Do we have a vtable?
        if i == 0 and base_class_offset > 0 and sbtype.IsPolymorphicClass():
            # We have a vtable at offset 0
            field_alignment = " " * ((depth + 1) * 4)
            field_type_name = "void *"
            field_name = "vtable"
            line = f"{field_alignment}+0x{base_offset:03x}"  # offset
            line += f" {color('poffsets_type')}{field_type_name:<10s}{color()}"  # type
            line += f" {field_name}"  # name
            print(line)

        print_sbtype_offset(
            base_class.GetType(), base_offset + base_class.GetOffsetInBytes(), depth + 1
        )

    # Fields
    for i in range(nb_fields):
        field = sbtype.GetFieldAtIndex(i)

        next_field_offset = base_offset + sbtype.GetByteSize()
        if i + 1 < nb_fields:
            next_field_offset = (
                base_offset + sbtype.GetFieldAtIndex(i + 1).GetOffsetInBytes()
            )

        print_field_offset(field, base_offset, depth, next_field_offset)

    print(f"{alignment}}};")


def cmd_poffsets(debugger, args, result, internal_dict):
    """Print class and structure offsets."""
    args = args.split()
    if len(args) < 1:
        print("Usage: poffsets <struct_name>")
        return

    struct_name = " ".join(args)
    dbg = Debugger(debugger)
    target = dbg.target()

    # FindFirstType doesn't found all the symbols
    structs = target.FindTypes(struct_name)
    if structs.GetSize() == 0:
        print(f"Error: element '{struct_name}' not found")
        return

    struct = structs.GetTypeAtIndex(0)
    print_sbtype_offset(struct, 0, 0)


################################################################################
# Command shared_cache -- Display information about a shared cache address


def search_shared_cache_region(target):
    process = target.GetProcess()
    regions = process.GetMemoryRegions()
    shared_cache_region = None
    region = lldb.SBMemoryRegionInfo()

    for i in range(regions.GetSize()):
        if not regions.GetMemoryRegionAtIndex(i, region):
            continue

        if region.GetRegionBase() < 0x180000000:
            # Not possible to be a shared cache address
            continue
        if not (
            region.IsReadable() and region.IsExecutable() and not region.IsWritable()
        ):
            continue

        shared_cache_region = region
        break

    return shared_cache_region


def shared_cache_images(dbg, shared_cache_region):
    """Return the shared cache images in the memory order"""
    shared_cache_base = shared_cache_region.GetRegionBase()
    shared_cache_end = shared_cache_region.GetRegionEnd()
    shared_cache_slide = shared_cache_base - 0x180000000

    shared_cache_size = shared_cache_end - shared_cache_base
    headers = Memory(
        dbg.read_memory(shared_cache_base, shared_cache_size), shared_cache_size
    )
    subcache_offset = headers.read32(0x1C0)
    subcache_array_cnt = headers.read32(0x1C4)

    modules = []

    for i in range(subcache_array_cnt):
        name_offset = headers.read32(subcache_offset + 0x20 * i + 0x18)
        module_name = headers.read_string(name_offset)

        module_offset = headers.read64(subcache_offset + 0x20 * i)
        module_addr = shared_cache_slide + module_offset
        modules.append({"begin": module_addr, "name": module_name})

    # Parsing all Mach-O could be costly so we use a simple heuristic here to know the end of each region
    modules.sort(key=lambda x: x["begin"])

    for i in range(len(modules) - 1):
        modules[i]["end"] = modules[i + 1]["begin"]

    region_info = lldb.SBMemoryRegionInfo()
    error = dbg.process().GetMemoryRegionInfo(modules[-1]["begin"], region_info)
    if error.Success():
        modules[-1]["end"] = region_info.GetRegionEnd()

    return modules


def macho_segments(dbg, base_addr):
    LC_SEGMENT_64 = 0x19
    mach_header = Memory(dbg.read_memory(base_addr, 0x20))  # mach_header_64
    ncmds = mach_header.read32(0x10)
    sizeofcmds = mach_header.read32(0x14)

    segments = {}

    offset = 32
    for _ in range(ncmds):
        if (offset - 32) > sizeofcmds:
            break

        cmd_struct = Memory(dbg.read_memory(base_addr + offset, 8))  # load_command
        cmd = cmd_struct.read32(0)
        cmdsize = cmd_struct.read32(4)

        if cmd == LC_SEGMENT_64:  # segment_command_64
            segment = Memory(dbg.read_memory(base_addr + offset, 0x48))
            seg_name = segment.read_string(0x08)
            seg_vmaddr = segment.read64(0x18)
            seg_vmsize = segment.read64(0x20)
            segments[seg_name] = {"addr": seg_vmaddr, "size": seg_vmsize}

        offset += cmdsize

    # Add the ASLR slide
    slide = base_addr - segments["__TEXT"]["addr"]
    for seg_name in segments.keys():
        segments[seg_name]["addr"] += slide

    return segments


def cmd_shared_cache(debugger, args, result, internal_dict):
    """Display information about the shared cache."""
    dbg = Debugger(debugger)
    target = dbg.target()

    # Parse the arguments
    usage = (
        "Usage:\n"
        "  shared_cache --info\n"
        "  shared_cache --all\n"
        "  shared_cache -a <addr>\n"
        "  shared_cache -i <image_name>"
    )

    args = args.split()
    if len(args) < 1:
        print(usage)
        return

    print_info = args[0] == "--info"
    find_all = args[0] == "--all"
    find_addr = args[0] == "-a"
    find_image = args[0] == "-i"
    if not (print_info or find_all or find_addr or find_image):
        print(usage)
        return

    addr_to_search = None
    if find_addr:
        if len(args) < 2:
            print(usage)
            return
        addr_to_search = dbg.expression(args[1]).GetValueAsUnsigned()

    image_to_search = None
    if find_image:
        if len(args) < 2:
            print(usage)
            return
        image_to_search = args[1]

    # Before any command, we need to find the shared cache base
    shared_cache_region = search_shared_cache_region(target)
    if not shared_cache_region:
        print("Error: shared cache base not found!")
        return

    shared_cache_base = shared_cache_region.GetRegionBase()
    shared_cache_end = shared_cache_region.GetRegionEnd()
    shared_cache_slide = shared_cache_base - 0x180000000

    # Show basic information
    if print_info:
        print(
            f"Shared cache header region: [0x{shared_cache_base:x} - 0x{shared_cache_end:x}]"
        )
        print(f"Shared cache slide:         0x{shared_cache_slide:x}")

        magic = dbg.read_string(shared_cache_base)
        print("Magic:                      " + magic)
        return

    # Parse the shared cache headers
    images = shared_cache_images(dbg, shared_cache_region)

    if find_all or find_image:
        for img in images:
            img_begin = img["begin"]
            img_end = img["end"]
            img_name = img["name"]

            if find_all:
                print(f"[0x{img_begin:x}-0x{img_end:x}] {img_name}")

            if find_image and (image_to_search.lower() in img_name.lower()):
                print(f"[0x{img_begin:x}-0x{img_end:x}] {img_name}")

        return

    # Parse the Mach-O segments for find_addr. This is a bit slow but acceptable
    for img in images:
        img_begin = img["begin"]
        img_end = img["end"]
        img_name = img["name"]
        segments = macho_segments(dbg, img_begin)

        for seg_name, seg in segments.items():
            seg_begin = seg["addr"]
            seg_end = seg_begin + seg["size"]

            if (addr_to_search >= seg_begin) and (addr_to_search < seg_end):
                print(f"0x{addr_to_search:x} information:")
                print(f"  Module:         [0x{img_begin:x}-0x{img_end:x}] {img_name}")
                module_offset = addr_to_search - img_begin
                print(f"  Module offset:  0x{module_offset:x}")

                segment_offset = addr_to_search - seg_begin
                print(f"  Segment:        [0x{seg_begin:x}-0x{seg_end:x}] {seg_name}")
                print(f"  Segment offset: 0x{segment_offset:x}")
                return


################################################################################
# Command telescope - Dereference the addresses


class Memory:
    """Deal with memory endianness"""

    def __init__(self, content, addr_size=8):
        self._mem = content
        self._addr_size = addr_size
        self._read_cb = {4: self.read32, 8: self.read64}

    def size(self):
        return len(self._mem)

    def read64(self, offset):
        return struct.unpack("<Q", self._mem[offset : offset + 8])[0]

    def read32(self, offset):
        return struct.unpack("<L", self._mem[offset : offset + 4])[0]

    def read8(self, offset):
        return self._mem[offset]

    def read_word(self, offset):
        if self._addr_size not in self._read_cb:
            raise Exception(
                "address size(in bytes) not supported: %#x", self._addr_size
            )
        return self._read_cb[self._addr_size](offset)

    def read_string(self, offset):
        str = ""
        while True:
            char = self.read8(offset)
            if not char:
                return str

            str += chr(char)
            offset += 1


def cmd_telescope(debugger, args, result, internal_dict):
    """Dereference the addresses."""
    args = args.split()
    if len(args) < 1:
        print("Usage: telescope <addr> [nb_ptr=8]")
        return

    dbg = Debugger(debugger)
    process = dbg.process()

    # lldb.SBTarget.addr_size: A read only property that returns the size in bytes of an address for this target.
    if process and process.target:
        addr_size = process.target.addr_size
    else:
        # default to 64-bits
        addr_size = 8

    base_addr = dbg.expression(args[0]).GetValueAsUnsigned()
    nb_ptr = 8
    if len(args) > 1:
        nb_ptr = int(args[1], 0)
        if nb_ptr <= 0:
            return

    try:
        memory = Memory(dbg.read_memory(base_addr, nb_ptr * addr_size), addr_size)
    except Exception as err:
        print(err)
        return

    # Prepare the address formatting
    max_offset_shown = (nb_ptr - 1) * 8
    nb_offset_digits = len(f"{max_offset_shown:x}")

    # Calculate all the lines
    details_list = []
    lines = []
    for i in range(nb_ptr):
        addr = base_addr + (i * addr_size)
        value = memory.read_word(addr - base_addr)

        details = dbg.details_for_addr(value)
        if details:
            details = "-> %s" % details

        if not details:
            # Maybe just a char
            if (value <= 126) and (value >= 33):
                details = "('%c')" % value

        details_list.append(details)
        addr_offset = f"0x%0{nb_offset_digits}x" % (i * addr_size)
        lines.append(
            f"{color('telescope_addr')}0x{addr:x}{color()}|+{addr_offset}: 0x{value:x}"
        )

    # Print the result. Do it in two times to align all the details
    max_line_len = max([len(x) for x in lines])
    for i in range(len(lines)):
        line = lines[i]
        details = details_list[i]
        if details:
            line += " " * (max_line_len + 1 - len(line))
            line += details
        print(line)


################################################################################
# Command xinfo -- Display information about an address


def cmd_xinfo(debugger, args, result, internal_dict):
    """Display information about an address."""
    args = args.split()
    if len(args) < 1:
        print("Usage: xinfo <addr>")
        return

    dbg = Debugger(debugger)
    target = dbg.target()
    addr = dbg.expression(args[0]).GetValueAsUnsigned()

    resolved_addr = dbg.target().ResolveLoadAddress(addr)
    details = dbg.details_for_addr(addr)
    if details:
        print(f"Symbol:      {details}")

    process = dbg.process()
    region_info = lldb.SBMemoryRegionInfo()
    error = process.GetMemoryRegionInfo(addr, region_info)
    if not error.Success():
        return

    region_base = region_info.GetRegionBase()
    region_end = region_info.GetRegionEnd()
    region_size = region_end - region_base
    print(
        f"Page:        0x{region_base:x} -> 0x{region_end:x} (size = 0x{region_size:x})"
    )

    mapped = "Yes" if region_info.IsMapped() else "No"
    print(f"Mapped:      {mapped}")

    permissions = ""
    permissions += "r" if region_info.IsReadable() else "-"
    permissions += "w" if region_info.IsWritable() else "-"
    permissions += "x" if region_info.IsExecutable() else "-"
    print(f"Permissions: {permissions}")

    module = resolved_addr.GetModule()
    if module:
        module_file_spec = module.GetFileSpec()
        module_path = (
            module_file_spec.GetDirectory() + "/" + module_file_spec.GetFilename()
        )
        print(f"Module:      {module_path}")

    # Get section for the address
    section = resolved_addr.GetSection()
    if section:
        section_name = section.GetName()
        print(f"Section:     {section_name}")


################################################################################
# Global code to handle commands, debugger, hooks, etc.


class CommandHandler:
    """Manage the LLDBX specific commands"""

    registered_cmd = {
        # Keep the commands in the alphabetical order
        "analyze": cmd_analyze,
        "br": cmd_br,
        "hexdump": cmd_hexdump,
        "conf": cmd_conf,
        "memset": cmd_memset,
        "memcpy": cmd_memcpy,
        "nopac": cmd_nopac,
        "poffsets": cmd_poffsets,
        "shared_cache": cmd_shared_cache,
        "telescope": cmd_telescope,
        "xinfo": cmd_xinfo,
    }

    @classmethod
    def print_usage(cls):
        lldbx_cmd = "lldbx"
        if REGISTER_COMMANDS:
            lldbx_cmd = "[lldbx]"  # optional

        print("Usage: %s <cmd> [args]\ncmd:" % lldbx_cmd)

        # Align the commands description
        biggest_name_len = max([len(name) for name in cls.registered_cmd])

        for name, method in cls.registered_cmd.items():
            print("    %-*s -- %s" % (biggest_name_len, name, method.__doc__))

    @classmethod
    def handle_cmd(cls, debugger, command, result, internal_dict):
        """Handle LLDBX commands."""
        cmd_name = None
        cmd_args = command.split()
        if len(cmd_args) > 0:
            cmd_name = cmd_args[0]
            cmd_args = cmd_args[1:]

        if cmd_name not in cls.registered_cmd:
            if cmd_name:
                print("ERR: Unknow command '%s'" % cmd_name)
            cls.print_usage()
            return

        method = cls.registered_cmd[cmd_name]
        method(debugger, " ".join(cmd_args), result, internal_dict)

    @classmethod
    def register(cls, debugger):
        cmd_doc = cls.handle_cmd.__doc__
        cmd_path = "%s.CommandHandler.handle_cmd" % __name__
        debugger.command(
            "command script add -o -h '%s' -f %s -- lldbx" % (cmd_doc, cmd_path)
        )

        if not REGISTER_COMMANDS:
            return

        for cmd_name in cls.registered_cmd:
            method = cls.registered_cmd[cmd_name]
            debugger.command(
                "command script add -o -h '%s' -f %s.%s %s"
                % (method.__doc__, __name__, method.__name__, cmd_name)
            )


class Debugger:
    """Debugger methods to be non-dependant to LLDB syntax"""

    def __init__(self, debugger):
        self._dbg = debugger

    def debugger(self):
        return self._dbg

    def target(self):
        return self._dbg.GetSelectedTarget()

    def process(self):
        target = self.target()
        if target:
            return target.GetProcess()
        return None

    def thread(self):
        process = self.process()
        if process:
            return process.GetSelectedThread()
        return None

    def frame(self):
        thread = self.thread()
        if thread:
            return thread.GetSelectedFrame()
        return None

    def expression(self, expr):
        target = self.target()
        if not target:
            return None
        return target.EvaluateExpression(expr)

    def command(self, cmd):
        if cmd:
            res = lldb.SBCommandReturnObject()
            ci = self._dbg.GetCommandInterpreter()
            ci.HandleCommand(cmd, res, False)
            if res.Succeeded():
                output = res.GetOutput()
                if output:
                    return output.strip()
                else:
                    return ""
            raise Exception(res.GetError().strip())
        raise Exception("No command specified")

    def disassemble(self, addr):
        target = self.target()
        resolved_addr = target.ResolveLoadAddress(addr)
        instructions = target.ReadInstructions(resolved_addr, 1)
        if len(instructions) < 1:
            return (None, None)

        mnemonic = instructions[0].GetMnemonic(target)
        operands = instructions[0].GetOperands(target)
        return (mnemonic, operands)

    def read_memory(self, addr, size):
        target = self.target()
        resolved_addr = target.ResolveLoadAddress(addr)
        error_ref = lldb.SBError()

        memory = target.ReadMemory(resolved_addr, size, error_ref)
        if not error_ref.Success() or not memory:
            raise Exception("Can't read memory at 0x%x" % addr)
        return memory

    def write_memory(self, addr, buffer):
        process = self.process()
        error_ref = lldb.SBError()

        nb_bytes_written = process.WriteMemory(addr, buffer, error_ref)
        if not error_ref.Success() or nb_bytes_written != len(buffer):
            raise Exception("Can't write memory at 0x%x" % addr)

    def can_read(self, addr, size=1):
        try:
            self.read_memory(addr, size)
            return True
        except:
            return False

    def read_string(self, addr, max_size=256):
        process = self.process()
        result = None

        error = lldb.SBError()
        try:
            cstring = process.ReadCStringFromMemory(addr, max_size, error)
            if error.Success() and cstring.isprintable():
                result = cstring
        except:
            pass  # Can fail if error is not set but no string was found

        return result

    def address_info(self, addr):
        info = {"is_readable": False, "cstring": None, "symbol": None, "offset": 0}
        info["is_readable"] = self.can_read(addr)
        if not info["is_readable"]:
            return info

        info["cstring"] = self.read_string(addr, 100)

        target = self.target()
        sb_addr = lldb.SBAddress(addr, target)
        sym_ctx = target.ResolveSymbolContextForAddress(
            sb_addr, lldb.eSymbolContextEverything
        )
        sym = None
        offset = 0
        if sym_ctx.GetSymbol().IsValid():
            sym = sym_ctx.GetSymbol().GetDisplayName()
            offset = addr - sym_ctx.GetSymbol().GetStartAddress().GetLoadAddress(target)
        elif sym_ctx.GetFunction().IsValid():
            sym = sym_ctx.GetFunction().GetDisplayName()
            offset = addr - sym_ctx.GetFunction().GetStartAddress().GetLoadAddress(
                target
            )

        info["symbol"] = sym
        info["offset"] = offset
        return info

    def _details_for_addr(self, addr):
        details = None
        addr_info = self.address_info(addr)

        if addr_info["cstring"]:
            details = "'%s'" % addr_info["cstring"]
        elif addr_info["symbol"]:
            symbol = addr_info["symbol"]
            offset = addr_info["offset"]
            details = "%s" % symbol
            if offset > 0:
                details += "+0x%x" % offset

        return details

    def details_for_addr(self, addr):
        user = nopac_user(addr)

        details = self._details_for_addr(addr)
        if not details and user != addr:
            details = self._details_for_addr(user)

        return details

    def resolve_adrp(self, opcode_addr, opcode):
        if (opcode & 0x9F000000) == 0x90000000:
            # adrp reg, offset
            offset = ((opcode & 0x60000000) >> 18) | ((opcode & 0xFFFFE0) << 8)
            target_addr = (offset << 1) + (opcode_addr & ~0xFFF)
            return target_addr
        return None

    def resolve_ldr_imm(self, opcode):
        if (opcode & 0xF9C00000) == 0xF9400000:
            # ldr reg, rn, imm
            imm = ((opcode >> 10) & 0xFFF) << 3
            return imm
        return None

    def resolve_add_imm(self, opcode):
        if (opcode & 0xFF000000) == 0x91000000:
            # add reg, rn, imm
            shift = (opcode >> 22) & 3
            imm = (opcode >> 10) & 0xFFF
            if shift == 1:
                imm <<= 12
            return imm
        return None

    def get_comment(self, target, inst):
        # By default take the LLDB comment
        comment = inst.GetComment(target)

        # Add more information if the previous operation is a ADRP address
        data = inst.GetData(target)
        err = lldb.SBError()
        opcode_addr = inst.GetAddress().GetLoadAddress(target)
        prev_opcode_addr = opcode_addr - 4

        memory = Memory(self.read_memory(prev_opcode_addr, 8))
        prev_opcode = memory.read32(0)
        opcode = memory.read32(4)

        target_addr = self.resolve_adrp(prev_opcode_addr, prev_opcode)
        if target_addr:
            imm_offset = self.resolve_ldr_imm(opcode)
            if imm_offset == None:
                imm_offset = self.resolve_add_imm(opcode)

            if imm_offset != None:
                addr_used = target_addr + imm_offset
                comment = hex(addr_used)
                details = self.details_for_addr(addr_used)

                if not details:
                    # We have an address but no symbol, try to resolve it again
                    try:
                        memory = Memory(self.read_memory(addr_used, 8))
                        addr_pointed = memory.read64(0)
                        details = self.details_for_addr(addr_pointed)
                        if details:
                            comment += " -> 0x%x" % addr_pointed
                    except:
                        # Can't read the address
                        pass

                if details:
                    comment += " (%s)" % details
                return comment

        return comment


class StopHook:
    """The target stop-hook registered inside LLDB"""

    def __init__(self, target, extra_args, dict):
        print("LLDBX stop-hook initialization")
        self._extra_args = extra_args
        debugger = target.GetDebugger()
        self._dbg = Debugger(debugger)
        self._prev_reg = {}

    def print_reg_values(self, exe_ctx):
        # Get registers values
        frame = exe_ctx.GetFrame()
        registers_list = frame.GetRegisters()
        registers = {}
        for category in registers_list:
            for reg in category:
                name = reg.GetName()
                value = reg.GetValueAsUnsigned()
                registers[name] = value

        # First stop, get all registers values
        if len(self._prev_reg) == 0:
            self._prev_reg = registers
            return

        # Save the new values
        updated_registers = {}
        blacklist = ["rip", "faultvaddr", "pc"]

        def has_registers(regs, name):
            # depends on arch: currently only support arm64
            vals = re.findall(r"^[wW](\d+)$", name)
            if vals and ((f"x{vals[0]}" in regs) or (f"X{vals[0]}" in regs)):
                return True
            return False

        def regs_from_operands(operands):
            # LLDB only give us the operands in string format. This is not
            # practiful but I don't want to use Capstone just for that. Do
            # something a bit better than grepping in the string otherwise "x2"
            # is a match for "x29".
            regs = set()
            reg = ""
            for charac in operands:
                if charac.isalnum():
                    reg += charac
                elif len(reg) > 0:
                    # Always use the same notation
                    if reg == "x29":
                        reg = "fp"
                    if reg == "x30":
                        reg = "lr"
                    if reg == "x31":
                        reg = "sp"
                    regs.add(reg)
                    reg = ""

            if len(reg) > 0:
                regs.add(reg)

            # Remove the numbers
            regs = [reg for reg in regs if not reg[0].isnumeric()]
            return regs

        for name, value in registers.items():
            if name in self._prev_reg:
                if name in blacklist:
                    continue
                prev_value = self._prev_reg[name]
                # can't do has_registers check here because I am not sure if there is any
                # ordering guarantee that X\d+ will come before W\d+
                if value != prev_value:
                    updated_registers[name] = value
                    self._prev_reg[name] = value

        # Get the registers used in the next instructions
        next_used_regs = {}
        _, next_operands = self._dbg.disassemble(frame.GetPC())
        next_registers = regs_from_operands(next_operands)

        for name, value in self._prev_reg.items():
            if name in next_registers:
                next_used_regs[name] = value

        # Show the updated registers
        for regs in list(updated_registers):
            if has_registers(updated_registers, regs):
                del updated_registers[regs]

        # Align all registers values
        max_len_name = 0
        all_regs = list(updated_registers.keys()) + list(next_used_regs.keys())
        for name in all_regs:
            if len(name) > max_len_name:
                max_len_name = len(name)

        for name, value in updated_registers.items():
            spaces = " " * (max_len_name - len(name))
            print(f"{color('reg_updated')}{name}{color()}{spaces} : 0x{value:x}")

        # Show the values which will be used
        for regs in list(next_used_regs):
            if (
                has_registers(next_used_regs, regs)
                or (regs in updated_registers)
                or has_registers(updated_registers, regs)
            ):
                del next_used_regs[regs]

        for name, value in next_used_regs.items():
            spaces = " " * (max_len_name - len(name))
            print(f"{color('reg_used')}{name}{color()}{spaces} : 0x{value:x}")

    def print_stack(self, exe_ctx):
        frame = exe_ctx.GetFrame()
        sp = frame.GetSP()
        args = "%s %d" % (sp, 8)
        cmd_telescope(self._dbg.debugger(), args, None, None)

    def print_seperator(self, name):
        width = 80
        separator = "-" * ((width - 4) - len(name))
        print("%s %s%s%s --" % (separator, color("separator"), name, color()))

    def print_code(self, exe_ctx):
        frame = exe_ctx.GetFrame()
        pc = frame.GetPC()
        target = self._dbg.target()

        resolved_addr = target.ResolveLoadAddress(pc)
        module_name = resolved_addr.GetModule().GetFileSpec().GetFilename()
        sym = resolved_addr.GetSymbol()
        instructions = sym.GetInstructions(target)

        inst_pos = -1
        i = 0
        for inst in instructions:
            if inst.GetAddress().GetLoadAddress(target) == pc:
                inst_pos = i
                break
            i += 1

        if inst_pos == -1:
            # Still show something
            print(self._dbg.command("x/5i $pc"))
            return

        first_inst_pos = max([0, inst_pos - 3])
        last_inst_pos = inst_pos + 5
        i = 0
        for inst in instructions:
            mnemonic = inst.GetMnemonic(target)
            operands = inst.GetOperands(target)
            comment = inst.GetComment(target)
            addr = inst.GetAddress().GetLoadAddress(target)
            if first_inst_pos <= i <= last_inst_pos:
                if i == first_inst_pos:
                    print("%s`%s:" % (module_name, sym.GetDisplayName()))

                spaces = "   "
                if i == inst_pos:
                    spaces = "-> "

                line = "%s0x%x: %-8s %s" % (spaces, addr, mnemonic, operands)
                line += " " * (80 - len(line))
                if len(comment) > 0:
                    line += " ; " + comment

                if i < inst_pos:
                    line = f"{color('code_prev')}{line}{color()}"
                elif i == inst_pos:
                    line = f"{color('code_current')}{line}{color()}"

                print(line)

            i += 1

    def handle_stop(self, exe_ctx, stream):
        # We can determine why we stop via exe_ctx.GetThread().GetStopReason()
        should_stop = True

        stop_thread = exe_ctx.GetThread()
        selected_thread = exe_ctx.GetTarget().GetProcess().GetSelectedThread()
        if stop_thread == selected_thread:
            self.print_seperator("registers")
            self.print_reg_values(exe_ctx)
            self.print_seperator("stack")
            self.print_stack(exe_ctx)
            self.print_seperator("code")
            self.print_code(exe_ctx)
            self.print_seperator("info")

        return should_stop


def __lldb_init_module(debugger, internal_dict):
    """Called when LLDB load the Python file"""

    dbg = Debugger(debugger)
    CommandHandler.register(dbg)

    # Remove any stop-hook already present to debug the commands
    # dbg.command("target stop-hook del")

    # Don't show anymore the source or the code
    dbg.command("settings set stop-disassembly-display never")
    dbg.command("settings set stop-line-count-before 0")
    dbg.command("settings set stop-line-count-after 0")

    # Register the stop-hook
    current_hooks = dbg.command("target stop-hook list")
    if "lldbx" not in current_hooks:
        dbg.command("target stop-hook add -P lldbx.StopHook")
