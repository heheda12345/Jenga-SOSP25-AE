# by chatgpt
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []

    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    return merged


def interval_intersection(intervals1, intervals2):
    i, j = 0, 0
    intersection = []

    while i < len(intervals1) and j < len(intervals2):
        start = max(intervals1[i][0], intervals2[j][0])
        end = min(intervals1[i][1], intervals2[j][1])

        if start <= end:
            intersection.append((start, end))

        if intervals1[i][1] < intervals2[j][1]:
            i += 1
        else:
            j += 1

    return intersection


def intersect_multiple_sets(sets):
    merged_sets = [merge_intervals(s) for s in sets]

    result = merged_sets[0]
    for s in merged_sets[1:]:
        result = interval_intersection(result, s)
        if not result:
            break

    return result


def to_range(bool_list):
    true_ranges = []
    start_idx = -1
    for i, b in enumerate(bool_list):
        if b and start_idx == -1:
            start_idx = i
        elif not b and start_idx != -1:
            true_ranges.append((start_idx, i - 1))
            start_idx = -1
    if start_idx != -1:
        true_ranges.append((start_idx, len(bool_list) - 1))
    return true_ranges


if __name__ == '__main__':
    sets = [[(1, 5), (10, 14), (16, 18)], [(2, 6), (8, 12), (15, 17)],
            [(3, 7), (9, 11), (13, 19)]]

    intersection = intersect_multiple_sets(sets)
    print(intersection)  # [(3, 5), (10, 11), (16, 17)]
