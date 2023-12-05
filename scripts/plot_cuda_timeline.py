import argparse
import pickle
from collections import OrderedDict
import re
import numpy as np
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


key_to_color_label = OrderedDict(
    {
        'call_forward': ('C0', 'forward'),
        'call_backward': ('C1', 'backward'),
        'cov_kron_A': ('C2', 'curvature'),
        'cov_kron_B': ('C2', None),
        'cov_unit_wise': ('C2', None),
        'inv_kron_A': ('C4', 'inverse'),
        'inv_kron_B': ('C4', None),
        'inv_unit_wise': ('C4', None),
        'sync_grad': ('C7', 'sync-grad'),
        'nb_sync_grad': ('C7', None),
        'reduce_scatter_grad': ('C7', None),
        'all_reduce_undivided_grad': ('C7', None),
        'all_gather_grad': ('C7', None),
        'all_reduce_no_curvature_grad': ('C7', None),
        'reduce_curvature': ('C9', 'sync-curvature'),
        'reduce_scatter_curvature': ('C9', 'sync-curvature'),
        'all_reduce_undivided_curvature': ('C9', None),
        'precondition': ('C8', 'precondition'),
    }
)


def sort(array, num_split):
    if num_split == 1:
        return array
    array_sorted = []
    for i in range(num_split):
        array_sorted += array[i:len(array):num_split]
    return array_sorted


def main():
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    min_time = timelines[0]['call_forward'][0][0]
    max_time = 0
    for start_end_list in timelines[0].values():
        for s, e in start_end_list:
            if e is not None:
                max_time = max(max_time, e)

    def time_shift(t):
        return (t - min_time) / 10 ** 6  # ns -> ms

    num_iterations = len(timelines[0]['start_end'])
    num_forward_per_iteration = len(timelines[0]['call_forward']) // num_iterations
    first_pipeline_time = time_shift(timelines[0]['call_forward'][num_forward_per_iteration][0])

    verts = []
    verts_alpha = []
    colors = []
    colors_alpha = []
    used_keys = set()
    width = .95
    usages = []
    for idx, timeline in enumerate(sort(timelines, args.num_replicas)):
        total_time_in_first_pipeline = 0
        y = len(timelines) - idx - 1
        for i, event_txt in enumerate(timeline):
            if not any(key in event_txt for key in key_to_color_label):
                continue
            key = next(key for key in key_to_color_label if key in event_txt)
            used_keys.add(key)
            start_end_list = timeline[event_txt]
            for s, e in start_end_list:
                if s is None or e is None:
                    continue
                s = time_shift(s)
                e = time_shift(e)
                if e < first_pipeline_time:
                    total_time_in_first_pipeline += e - s
                v = [(s, y-width/2), (s, y+width/2), (e, y+width/2), (e, y-width/2), (s, y-width/2)]
                print(v, event_txt)
                color, label = key_to_color_label[key]
                if any(keyword in key for keyword in ['sync', 'reduce', 'gather']):
                    verts_alpha.append(v)
                    colors_alpha.append(color)
                else:
                    verts.append(v)
                    colors.append(color)
        usages.append(total_time_in_first_pipeline / first_pipeline_time)
    usage = np.mean(usages)

    bars = PolyCollection(verts, facecolors=colors)
    ax.add_collection(bars)
    bars = PolyCollection(verts_alpha, facecolors=colors_alpha, alpha=.5, hatch='//')
    ax.add_collection(bars)
    ax.autoscale()

    ax.set_xlabel('Time (ms)')
    ax.set_yticks(range(len(timelines)))
    ax.set_yticklabels([f'GPU {i+1}' for i in range(len(timelines))][::-1])
    ax.set_title(f'{args.title} [GPU util. {usage * 100:.1f}%]')
    ax.set_xlim(time_shift(min_time), time_shift(max_time))

    durations = []
    prev_start = None
    for i in range(1, num_iterations):
        start_time = time_shift(timelines[0]['call_forward'][num_forward_per_iteration * i][0])
        if prev_start is None:
            durations.append(start_time)
        else:
            durations.append(start_time - prev_start)
        prev_start = start_time
        ax.axvline(start_time, color='r', lw=7, label='flush @ GPU1' if i == 1 else None)
    print('avg duration', np.mean(durations))
    for key, (color, label) in key_to_color_label.items():
        if key in used_keys:
            if any(keyword in key for keyword in ['sync', 'reduce', 'gather']):
                ax.bar(0, 0, label=label, color=color, alpha=0.5, hatch='//')
            else:
                ax.bar(0, 0, label=label, color=color)
#    if len(used_keys) + 1 > 6:
#        ax.legend(bbox_to_anchor=(0, 1.2), loc='upper left', ncol=6)
#    else:
#        ax.legend(bbox_to_anchor=(0, 1.15), loc='upper left', ncol=len(used_keys)+1)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def human_sort(l):
    """
    Sort a list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_paths', type=str)
    parser.add_argument('--fig_path', type=str, default='prof.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--num_replicas', type=int, default=1)
    args = parser.parse_args()

    timelines = []
    pickle_paths = args.pickle_paths.split(',')
    human_sort(pickle_paths)
    for pickle_path in pickle_paths:
        if pickle_path == '':
            continue
        timelines.append(pickle.load(open(pickle_path, 'rb')))
    main()
