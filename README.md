# lldbx

LLDB eXtended for macOS & iOS


## Overview

lldbx is an extended interface for LLDB, designed to enhance debugging workflows for macOS and iOS developers. It provides additional commands, improved output formatting, and automation features to streamline the debugging process.


## Features

- Enhanced LLDB commands for common debugging tasks
- Improved output readability and formatting
- Automation for repetitive debugging steps
- Support for macOS and iOS targets
- Easy integration with existing LLDB workflows

```
(lldb) lldbx
Usage: [lldbx] <cmd> [args]
cmd:
    analyze      -- Analyze a function and show a more readable disassembled version.
    br           -- Add a breakpoint on an address or a symbol.
    hexdump      -- Hexadecimal dump.
    conf         -- Configure global settings
    memset       -- memset equivalent primitive.
    memcpy       -- memcpy equivalent primitive.
    nopac        -- Remove PAC from a pointer.
    poffsets     -- Print class and structure offsets.
    shared_cache -- Display information about the shared cache.
    telescope    -- Dereference the addresses.
    xinfo        -- Display information about an address.
```


## Installation

```sh
cd
wget https://raw.githubusercontent.com/DimitriFourny/lldbx/refs/heads/main/lldbx.py
echo 'command script import "~/lldbx.py"' > ~/.lldbinit
```


## Usage

- Launch LLDB and load lldbx extensions as described in the documentation.
- Use the enhanced commands to debug your macOS or iOS applications more efficiently.


## Documentation

See the [lldbx.html](lldbx.html) file for detailed documentation, command reference, and examples.
