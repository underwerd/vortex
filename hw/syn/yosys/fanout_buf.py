#!/usr/bin/env python3
"""fanout_buf.py — insert BUF_X trees on high-fanout nets of a mapped netlist.

yosys+ABC does not buffer high-fanout nets (math-sfu-002 V4b: the math-lane
clock-enable net reached 2330 loads and slew-collapsed to >150 ns of NLDM-
extrapolated delay at 400 MHz).  This post-processor rewrites the gate-level
netlist that `abc` produces: every net with more than --threshold load pins
gets a balanced BUF_X{4,8,16,32} tree, and each load pin is re-bound to a
leaf.  Clock/reset nets are skipped.

Netlist contract (yosys `write_verilog -noattr -noexpr`):
  - one cell instance per statement, pins one per line, nets carry no parens
  - cell type must exist in the liberty file passed via --liberty
  - escaped identifiers (\\name, terminated by whitespace) are supported

Usage: fanout_buf.py netlist.v out.v --liberty lib.typ --threshold 48
"""
import argparse
import re
import sys

OUT_PINS = {'Q', 'QN', 'Z', 'ZN', 'X', 'Y', 'CO'}
# cells whose .S pin is an output (full/half adders); everywhere else .S is
# the mux select input and must stay a load
S_OUTPUT_CELLS = {'FA_X1', 'HA_X1'}

NAME = r'(?:\\[^\s]+\s|[A-Za-z_][A-Za-z0-9_$]*)'


def lib_cells(lib_path):
    cells = set()
    with open(lib_path) as f:
        for line in f:
            m = re.match(r'\s*cell\s*\(\s*([A-Za-z0-9_]+)\s*\)', line)
            if m:
                cells.add(m.group(1))
    return cells


def split_pins(body):
    pins = []
    i, n = 0, len(body)
    while i < n:
        m = re.match(r'\s*\.([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?)\s*\(', body[i:])
        if not m:
            if re.match(r'\s*$', body[i:]):
                break
            raise ValueError(f'bad pin near {body[i:i+60]!r}')
        pin = m.group(1)
        i += m.end()
        if body[i] == '\\':
            j = i + 1
            while j < n and body[j] != ' ':
                j += 1
            net = body[i:j]
            i = j
        else:
            m2 = re.match(r'[A-Za-z_][A-Za-z0-9_$]*(?:\[[0-9]+\])?', body[i:])
            if not m2:
                raise ValueError(f'bad net near {body[i:i+60]!r}')
            net = m2.group(0)
            i += m2.end()
        m3 = re.match(r'\s*\)\s*,?\s*', body[i:])
        if not m3:
            raise ValueError(f'bad pin tail near {body[i:i+60]!r}')
        i += m3.end()
        pins.append((pin, net))
    return pins


def parse_instances(text, cells):
    """[(start, end, cell, inst, [(pin, net)])] for each instance statement."""
    out = []
    for m in re.finditer(r'^\s*([A-Za-z0-9_]+)\s+(%s)\(' % NAME, text, re.M):
        cell = m.group(1)
        if cell not in cells:
            continue
        inst = m.group(2)
        body_start = m.end()
        depth, j = 1, m.end()
        while depth:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
            j += 1
        out.append((m.start(), j, cell, inst, split_pins(text[body_start:j - 1])))
    return out


def tree_shape(fanout):
    """Levels root->leaves as (buf_size, count).  Leaves drive <= 24 loads,
    every other buffer drives <= 16 buffer inputs."""
    LEAF_MAX, BRANCH = 24, 16
    sizes = {0: 4}                      # level index from leaves -> cell size
    counts = [ (fanout + LEAF_MAX - 1) // LEAF_MAX ]
    while counts[-1] > 1:
        counts.append((counts[-1] + BRANCH - 1) // BRANCH)
    levels = []
    for li, cnt in enumerate(reversed(counts)):
        depth_from_leaf = len(counts) - 1 - li
        size = {0: 4, 1: 8, 2: 16}.get(depth_from_leaf, 32)
        levels.append((size, cnt))
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--liberty', required=True)
    ap.add_argument('--threshold', type=int, default=48)
    ap.add_argument('--skip', action='append', default=[])
    args = ap.parse_args()

    cells = lib_cells(args.liberty)
    text = open(args.infile).read()
    insts = parse_instances(text, cells)

    loads = {}
    for _, _, cell, _, pins in insts:
        for pin, net in pins:
            is_out = pin.split('[')[0] in OUT_PINS or (
                pin.split('[')[0] == 'S' and cell in S_OUTPUT_CELLS)
            if not is_out:
                loads[net] = loads.get(net, 0) + 1

    skip = set(args.skip) | {'clk', 'reset', 'resetn'}
    targets = sorted((n for n, c in loads.items()
                      if c > args.threshold and n not in skip),
                     key=lambda n: -loads[n])
    for n in targets:
        print(f'# buffer-tree target: fanout={loads[n]}  net={n[:120]}',
              file=sys.stderr)
    if not targets:
        open(args.outfile, 'w').write(text)
        print('# no high-fanout nets; netlist copied unchanged', file=sys.stderr)
        return

    # build trees: for each target, level nets root->leaves
    buf_cells = []                      # (size, parent_net, out_net)
    leaf_of = {}
    buf_id = 0
    for net in targets:
        levels = tree_shape(loads[net])
        prev = [net]
        for li, (size, cnt) in enumerate(levels):
            nxt = []
            for k in range(cnt):
                oname = f'fbuf_{buf_id}_z'
                buf_id += 1
                buf_cells.append((size, prev[min(k, len(prev) - 1)], oname))
                nxt.append(oname)
            prev = nxt
        leaf_of[net] = prev

    # re-bind load pins round-robin onto leaves
    pieces, last = [], 0
    leaf_idx = {n: 0 for n in targets}
    for start, end, cell, inst, pins in insts:
        new_pins, changed = [], False
        for pin, net in pins:
            base = pin.split('[')[0]
            is_out = base in OUT_PINS or (base == 'S' and cell in S_OUTPUT_CELLS)
            if net in leaf_of and not is_out:
                leaves = leaf_of[net]
                new_pins.append((pin, leaves[leaf_idx[net] % len(leaves)]))
                leaf_idx[net] += 1
                changed = True
            else:
                new_pins.append((pin, net))
        if changed:
            pieces.append(text[last:start])
            body = ',\n    '.join(f'.{p}({n})' for p, n in new_pins)
            pieces.append(f'  {cell} {inst} (\n    {body}\n  );\n')
            last = end
    pieces.append(text[last:])
    text = ''.join(pieces)

    buf_lines = [f'  BUF_X{size} fbuf_cell_{i} (\n    .A({parent}),\n    .Z({oname})\n  );\n'
                 for i, (size, parent, oname) in enumerate(buf_cells)]
    if buf_lines:
        text = text.replace('endmodule', ''.join(buf_lines) + 'endmodule', 1)
    open(args.outfile, 'w').write(text)
    print(f'# inserted {len(buf_cells)} buffers for {len(targets)} nets',
          file=sys.stderr)


if __name__ == '__main__':
    main()
