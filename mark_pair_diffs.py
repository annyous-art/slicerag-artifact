#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from difflib import SequenceMatcher


def _changed_line_numbers(raw: object) -> list[int]:
    if not isinstance(raw, dict):
        return []

    line_numbers: list[int] = []
    for key in raw.keys():
        try:
            line_numbers.append(int(key))
        except (TypeError, ValueError):
            continue
    line_numbers.sort()
    return line_numbers


def load_diff_lookup(diff_path: Path):
    deleted_lookup: dict[int, list[int]] = {}
    added_lookup: dict[int, list[int]] = {}

    if not diff_path.exists():
        return deleted_lookup, added_lookup

    with diff_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            idx_vul = item.get('idx_vul')
            idx_novul = item.get('idx_novul')
            if idx_vul is not None:
                deleted_lookup[int(idx_vul)] = _changed_line_numbers(item.get('deleted_lines'))
            if idx_novul is not None:
                added_lookup[int(idx_novul)] = _changed_line_numbers(item.get('added_lines'))

    return deleted_lookup, added_lookup


def _sequence_labels(lines_a, lines_b):
    sm = SequenceMatcher(None, lines_a, lines_b)
    labels_a = [0] * len(lines_a)
    labels_b = [0] * len(lines_b)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        for k in range(i1, i2):
            labels_a[k] = 1
        for k in range(j1, j2):
            labels_b[k] = 1
    return labels_a, labels_b


def find_multiline_indices(lines):
    """Find line indices that are inside /* ... */ or <!-- ... --> blocks (inclusive)."""
    idxs = set()
    in_block = False
    start_mark = None
    for i, ln in enumerate(lines):
        s = ln
        s_strip = s.strip()
        if not in_block and (s_strip.startswith('/*') or s_strip.startswith('<!--')):
            in_block = True
            start_mark = '*/' if s_strip.startswith('/*') else '-->'
            idxs.add(i)
            if (start_mark == '*/' and '*/' in s_strip and s_strip.index('/*') <= s_strip.index('*/')) or (start_mark == '-->' and '-->' in s_strip and s_strip.index('<!--') <= s_strip.index('-->')):
                in_block = False
                start_mark = None
            continue
        if in_block:
            idxs.add(i)
            if start_mark and start_mark in s_strip:
                in_block = False
                start_mark = None
    return idxs


def _is_ignorable_line(s: str) -> bool:
    """Check if a line is ignorable (empty, brace, comment marker)."""
    s_strip = s.strip()
    if not s_strip:
        return True
    if s_strip in ('{', '}', '};'):
        return True
    if s_strip.startswith('//'):
        return True
    if s_strip in ('/*', '*/', '<!--', '-->'):
        return True
    return False


def _addition_only_protected_lines(old_lines, new_lines, max_expand=10):
    """For addition-only fixes, find lines in the vulnerable version that the fix is protecting.

    When a fix only ADDS code (bounds checks, null checks, etc.) without deleting
    or modifying existing lines, the vulnerable version has no "changed" lines to mark.
    This function uses SequenceMatcher to align old and new code, finds the insertion
    points, and marks the lines in the vulnerable version that are adjacent to or
    protected by the inserted code.

    Returns sorted list of 0-indexed line positions in old_lines to mark as label=1.
    """
    sm = SequenceMatcher(None, old_lines, new_lines)
    candidates = set()

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            # Code was inserted before old[i1]; old[i1] is the "protected" line
            if i1 < len(old_lines):
                candidates.add(i1)
            # old[i1-1] provides context before the insertion
            if i1 - 1 >= 0:
                candidates.add(i1 - 1)
        elif tag == 'replace':
            # Replaced lines in the old version are also relevant
            for k in range(i1, i2):
                candidates.add(k)

    # Filter out ignorable lines and lines inside multiline comment blocks
    multiline = find_multiline_indices(old_lines)
    protected = set()
    for idx in candidates:
        if 0 <= idx < len(old_lines) and idx not in multiline and not _is_ignorable_line(old_lines[idx]):
            protected.add(idx)

    # If all candidates were filtered out (e.g. all braces/comments),
    # expand outward to find the nearest non-ignorable, non-multiline line
    if not protected:
        for idx in sorted(candidates):
            for delta in range(1, max_expand + 1):
                for c in [idx - delta, idx + delta]:
                    if 0 <= c < len(old_lines) and c not in multiline and not _is_ignorable_line(old_lines[c]):
                        protected.add(c)
                        break
                if protected:
                    break

    return sorted(protected)



# def compute_labels(lines_a, lines_b):
#     sm = SequenceMatcher(None, lines_a, lines_b)
#     labels_a = [0] * len(lines_a)
#     labels_b = [0] * len(lines_b)
#     for tag, i1, i2, j1, j2 in sm.get_opcodes():
#         if tag == 'equal':
#             continue
#         # replace, delete, insert -> mark affected lines as 1
#         for k in range(i1, i2):
#             labels_a[k] = 1
#         for k in range(j1, j2):
#             labels_b[k] = 1

#     def is_brace_or_empty(s: str) -> bool:
#         s_strip = s.strip()
#         # Filter only brace-only lines, not lines containing code.
#         return s_strip == '' or s_strip in ('{', '}', '};', '{,', '},')

#     suppressed_a = 0
#     suppressed_b = 0
#     for idx, line in enumerate(lines_a):
#         if labels_a[idx] == 1 and is_brace_or_empty(line):
#             labels_a[idx] = 0
#             suppressed_a += 1
#     for idx, line in enumerate(lines_b):
#         if labels_b[idx] == 1 and is_brace_or_empty(line):
#             labels_b[idx] = 0
#             suppressed_b += 1

#     return labels_a, labels_b, suppressed_a, suppressed_b


def compute_labels(lines_a, lines_b, changed_lines_a=None, changed_lines_b=None):
    if changed_lines_a is not None and changed_lines_b is not None:
        labels_a = [0] * len(lines_a)
        labels_b = [0] * len(lines_b)
        for line_no in changed_lines_a:
            idx = line_no - 1
            if 0 <= idx < len(labels_a):
                labels_a[idx] = 1
        for line_no in changed_lines_b:
            idx = line_no - 1
            if 0 <= idx < len(labels_b):
                labels_b[idx] = 1
    else:
        labels_a, labels_b = _sequence_labels(lines_a, lines_b)

    # --- Addition-only fix handling ---
    # When the vuln side has 0 changed lines but the non-vuln side has additions,
    # the fix added new code (bounds checks, null checks, etc.) without modifying
    # existing lines. Use SequenceMatcher to find lines in the vulnerable version
    # that the fix is protecting/relating to.
    addition_only_info = None
    if changed_lines_a is not None and len(changed_lines_a) == 0 and len(changed_lines_b) > 0:
        protected = _addition_only_protected_lines(lines_a, lines_b)
        marked = [idx for idx in protected if 0 <= idx < len(labels_a)]
        addition_only_info = {
            'protected_indices': marked,
            'protected_lines': [lines_a[idx][:80] for idx in marked if idx < len(lines_a)],
        }
        for idx in protected:
            if 0 <= idx < len(labels_a):
                labels_a[idx] = 1

    suppressed_info_a = []
    suppressed_info_b = []
    # mark lines that are inside multiline-comment blocks so inner lines are also ignorable
    multiline_a = find_multiline_indices(lines_a)
    multiline_b = find_multiline_indices(lines_b)
    for idx, line in enumerate(lines_a):
        if labels_a[idx] == 1:
            if idx in multiline_a:
                suppressed_info_a.append((idx, line, 'multiline_content'))
                labels_a[idx] = 0
                continue
            if _is_ignorable_line(line):
                # Determine reason for backward compatibility in suppressed entries
                s_strip = line.strip()
                if not s_strip:
                    reason = 'empty'
                elif s_strip in ('{', '}', '};'):
                    reason = 'brace'
                elif s_strip.startswith('//'):
                    reason = 'line_comment'
                elif s_strip in ('/*', '*/', '<!--', '-->'):
                    reason = 'multiline_marker'
                else:
                    reason = 'other'
                suppressed_info_a.append((idx, line, reason))
                labels_a[idx] = 0
    for idx, line in enumerate(lines_b):
        if labels_b[idx] == 1:
            if idx in multiline_b:
                suppressed_info_b.append((idx, line, 'multiline_content'))
                labels_b[idx] = 0
                continue
            if _is_ignorable_line(line):
                s_strip = line.strip()
                if not s_strip:
                    reason = 'empty'
                elif s_strip in ('{', '}', '};'):
                    reason = 'brace'
                elif s_strip.startswith('//'):
                    reason = 'line_comment'
                elif s_strip in ('/*', '*/', '<!--', '-->'):
                    reason = 'multiline_marker'
                else:
                    reason = 'other'
                suppressed_info_b.append((idx, line, reason))
                labels_b[idx] = 0

    return labels_a, labels_b, suppressed_info_a, suppressed_info_b, addition_only_info


def process_file(path: Path, deleted_lookup: dict[int, list[int]], added_lookup: dict[int, list[int]]):
    # use stem + '_labeled' + suffix, e.g. foo.jsonl -> foo_labeled.jsonl
    out_path = path.with_name(path.stem + '_labeled' + path.suffix)
    total_pairs = 0
    updated = 0
    diff_lines = 0
    total_suppressed = 0
    suppressed_entries = []
    addition_only_count = 0
    addition_only_surviving = 0
    # run statistics for contiguous 1 segments
    file_total_runs = 0
    file_total_run_lines = 0
    file_objects_with_runs = 0
    file_sum_per_object_avg_run = 0.0
    with path.open('r', encoding='utf-8') as f_in, out_path.open('w', encoding='utf-8') as f_out:
        lines = f_in.read().splitlines()
        # parse json objects per line
        objs = [json.loads(l) for l in lines if l.strip()]
        n = len(objs)
        i = 0
        while i + 1 < n:
            a = objs[i]
            b = objs[i+1]
            total_pairs += 1
            func_a = a.get('func', '') or ''
            func_b = b.get('func', '') or ''
            la = func_a.splitlines()
            lb = func_b.splitlines()
            changed_lines_a = None
            changed_lines_b = None
            try:
                idx_a = int(a.get('idx')) if a.get('idx') is not None else None
            except (TypeError, ValueError):
                idx_a = None
            try:
                idx_b = int(b.get('idx')) if b.get('idx') is not None else None
            except (TypeError, ValueError):
                idx_b = None

            if idx_a is not None:
                changed_lines_a = deleted_lookup.get(idx_a) if int(a.get('target', -1)) == 1 else added_lookup.get(idx_a)
            if idx_b is not None:
                changed_lines_b = deleted_lookup.get(idx_b) if int(b.get('target', -1)) == 1 else added_lookup.get(idx_b)

            labels_a, labels_b, suppressed_info_a, suppressed_info_b, addition_only_info = compute_labels(
                la,
                lb,
                changed_lines_a=changed_lines_a,
                changed_lines_b=changed_lines_b,
            )
            # accumulate diff lines (both sides)
            diff_lines += sum(labels_a) + sum(labels_b)
            # accumulate suppressed counts and collect details for the whole file
            total_suppressed += len(suppressed_info_a) + len(suppressed_info_b)
            # track addition-only fix statistics
            if addition_only_info is not None:
                addition_only_count += 1
                surviving = sum(1 for idx in addition_only_info['protected_indices'] if labels_a[idx] == 1)
                addition_only_surviving += surviving
            # record suppressed entries with object/global context
            # prefer to use the object's own `idx` field if present, otherwise fall back to the file index
            obj_idx_a = a.get('idx', i)
            obj_idx_b = b.get('idx', i + 1)
            for (li, line_text, reason) in suppressed_info_a:
                suppressed_entries.append({'object_index': obj_idx_a, 'side': 'a', 'line_index': li, 'line': line_text, 'reason': reason})
            for (li, line_text, reason) in suppressed_info_b:
                suppressed_entries.append({'object_index': obj_idx_b, 'side': 'b', 'line_index': li, 'line': line_text, 'reason': reason})
            # compute contiguous-run stats per object
            def runs_from_labels(lbls):
                runs = []
                cur = 0
                for v in lbls:
                    if v == 1:
                        cur += 1
                    else:
                        if cur > 0:
                            runs.append(cur)
                            cur = 0
                if cur > 0:
                    runs.append(cur)
                return runs

            runs_a = runs_from_labels(labels_a)
            runs_b = runs_from_labels(labels_b)
            # update file-level run stats
            file_total_runs += len(runs_a) + len(runs_b)
            file_total_run_lines += sum(runs_a) + sum(runs_b)
            if len(runs_a) > 0:
                file_objects_with_runs += 1
                file_sum_per_object_avg_run += (sum(runs_a) / len(runs_a))
            if len(runs_b) > 0:
                file_objects_with_runs += 1
                file_sum_per_object_avg_run += (sum(runs_b) / len(runs_b))

            # attach labels to each object (aligned to its own function lines)
            a['labels'] = labels_a
            b['labels'] = labels_b
            # write them back (preserve order)
            f_out.write(json.dumps(a, ensure_ascii=False) + '\n')
            f_out.write(json.dumps(b, ensure_ascii=False) + '\n')
            updated += 2
            i += 2
        # if odd last object, write it unchanged (no pair)
        if i < n:
            f_out.write(json.dumps(objs[i], ensure_ascii=False) + '\n')
    return out_path, total_pairs, updated, diff_lines, total_suppressed, suppressed_entries, file_total_runs, file_total_run_lines, file_objects_with_runs, file_sum_per_object_avg_run, addition_only_count, addition_only_surviving


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Mark pair diffs and optionally process a single file')
    parser.add_argument('--file', '-f', help='Path to a single .jsonl file to process (absolute or relative)')
    parser.add_argument('--diff-file', default=Path(__file__).with_name('diff_lines.jsonl'), help='Path to diff_lines.jsonl')
    parser.add_argument('--dump-suppressed', help='path to write suppressed details (jsonl)')
    args = parser.parse_args()

    deleted_lookup, added_lookup = load_diff_lookup(Path(args.diff_file))

    # decide which files to process
    if args.file:
        p = Path(args.file)
        if not p.exists() or not p.is_file():
            print(f'File not found: {args.file}')
            return
        files = [p]
    else:
        cwd = Path.cwd()
        files = sorted(
            p for p in cwd.glob('*.jsonl')
            if p.name.endswith('_paired.jsonl') and not p.name.endswith('_paired_labeled.jsonl')
        )
        if not files:
            print('No .jsonl files found in', cwd)
            return

    total_pairs_all = 0
    total_objects = 0
    total_diff_lines = 0
    total_suppressed_all = 0
    total_runs_all = 0
    total_run_lines_all = 0
    total_objects_with_runs_all = 0
    total_sum_per_object_avg_run_all = 0.0
    addition_only_count_all = 0
    addition_only_surviving_all = 0
    dump_path_arg = args.dump_suppressed

    for p in files:
        (out_path, pairs, updated, diff_lines, suppressed, suppressed_entries,
         file_total_runs, file_total_run_lines, file_objects_with_runs, file_sum_per_object_avg_run,
         addition_only_count, addition_only_surviving) = process_file(
            p,
            deleted_lookup,
            added_lookup,
        )
        # pairs -> number of pairs; updated -> written records (objects)
        total_pairs_all += pairs
        total_objects += updated
        total_diff_lines += diff_lines
        total_suppressed_all += suppressed
        addition_only_count_all += addition_only_count
        addition_only_surviving_all += addition_only_surviving
        # accumulate run stats
        total_runs_all += file_total_runs
        total_run_lines_all += file_total_run_lines
        total_objects_with_runs_all += file_objects_with_runs
        total_sum_per_object_avg_run_all += file_sum_per_object_avg_run
        # optionally dump suppressed entries to file (default into output/ per-file)
        # determine dump path per-file
        if dump_path_arg:
            dump_path = Path(dump_path_arg)
        else:
            dump_dir = Path('output')
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f'suppressed_{p.stem}.jsonl'

        if suppressed_entries:
            # write JSONL (overwrite per run)
            with open(dump_path, 'w', encoding='utf-8') as df:
                for entry in suppressed_entries:
                    out = {'file': str(p), **entry}
                    df.write(json.dumps(out, ensure_ascii=False) + '\n')
            # also write TSV alongside JSONL for convenience
            tsv_path = dump_path.with_suffix('.tsv')
            import csv
            with open(tsv_path, 'w', encoding='utf-8', newline='') as tf:
                writer = csv.writer(tf, delimiter='\t')
                writer.writerow(['file','object_index','side','line_index','reason','line'])
                for entry in suppressed_entries:
                    writer.writerow([str(p), entry.get('object_index'), entry.get('side'), entry.get('line_index'), entry.get('reason'), entry.get('line')])
        print(f'Processed {p.name}: pairs={pairs}, written_records={updated}, diff_lines={diff_lines}, suppressed={suppressed}, runs={file_total_runs}, run_lines={file_total_run_lines}, objects_with_runs={file_objects_with_runs}, out={out_path}')

    num_files = len(files)
    avg_diff_per_object = total_diff_lines / total_objects if total_objects else 0.0
    avg_run_length_overall = (total_run_lines_all / total_runs_all) if total_runs_all else 0.0
    avg_run_length_per_object_including_zero = (total_run_lines_all / total_objects) if total_objects else 0.0
    avg_run_length_per_object_excluding_zero = (total_sum_per_object_avg_run_all / total_objects_with_runs_all) if total_objects_with_runs_all else 0.0

    print('\nSummary across processed files:')
    print(f'Files processed: {num_files}')
    print(f'Total pairs: {total_pairs_all}')
    print(f'Total objects (written records): {total_objects}')
    print(f'Total diff lines (both sides): {total_diff_lines}')
    print(f'Total suppressed by is_brace_or_empty: {total_suppressed_all}')
    print(f'Total runs (contiguous 1 segments): {total_runs_all}')
    print(f'Total run lines (sum of run lengths): {total_run_lines_all}')
    print(f'Average run length (overall, per run): {avg_run_length_overall:.3f}')
    print(f'Average run length per object (including zero-run objects): {avg_run_length_per_object_including_zero:.3f}')
    print(f'Average run length per object (excluding zero-run objects): {avg_run_length_per_object_excluding_zero:.3f}')
    print(f'Average diff lines per object: {avg_diff_per_object:.3f}')
    print(f'Addition-only fixes with new labels: {addition_only_count_all}')
    print(f'Addition-only surviving label-1 lines (after suppression): {addition_only_surviving_all}')


if __name__ == '__main__':
    main()
