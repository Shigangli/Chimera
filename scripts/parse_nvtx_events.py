import argparse
import sqlite3
import warnings
import statistics
import pickle

import pandas as pd


def get_all_event_texts():
    sql = f"""
    SELECT DISTINCT text
    FROM NVTX_EVENTS;
    """
    df = pd.read_sql(sql, con)
    return [row['text'] for _, row in df.iterrows()]


def get_event_start_end(event_text):
    sql = f"""
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = '{event_text}';
    """
    df = pd.read_sql(sql, con)
    return [(row['start'], row['end']) for _, row in df.iterrows()]


def get_total_time_in_event(target_table_name, event_start, event_end):
    sql = f"""
    SELECT SUM(target.end - target.start) AS total_time
    FROM {target_table_name} target
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime
      ON target.correlationId = runtime.correlationId
    WHERE runtime.start BETWEEN {event_start} AND {event_end};
    """
    df = pd.read_sql(sql, con)
    time = df['total_time'].iloc[0]
    if time is None:
        return 0
    return time


def get_start_end_in_event(target_table_name, event_start, event_end):
    sql = f"""
    SELECT MIN(target.start), MAX(target.end)
    FROM {target_table_name} target
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime
      ON target.correlationId = runtime.correlationId
    WHERE runtime.start BETWEEN {event_start} AND {event_end};
    """
    df = pd.read_sql(sql, con)
    start = df['MIN(target.start)'].iloc[0]
    end = df['MAX(target.end)'].iloc[0]
    return start, end


def get_runtime_in_event(event_start, event_end):
    return event_end - event_start


def get_kernel_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_KERNEL', event_start, event_end)


def get_kernel_start_end_in_event(event_start, event_end):
    return get_start_end_in_event('CUPTI_ACTIVITY_KIND_KERNEL', event_start, event_end)


def get_memset_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_MEMSET', event_start, event_end)


def get_memcpy_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_MEMCPY', event_start, event_end)


def get_sync_time_in_event(event_start, event_end):
    return get_total_time_in_event('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', event_start, event_end)


def get_stats(event_texts, pickle_path):
    times = {'ncalls': []}
    for key in ['runtime', 'kernel', 'memset', 'memcpy', 'sync']:
        times[key] = []
        times[f'{key}_stdev'] = []
    index = []
    print(f'Collecting time for {event_texts}')
    for txt in event_texts:
        event_start_end = get_event_start_end(txt)
        if len(event_start_end) == 0:
            continue
        index.append(txt)
        if args.ignore_first_event:
            # ignore first NVTX event
            event_start_end = event_start_end[1:]
        times['ncalls'].append(len(event_start_end))
        for key, f in {'runtime': get_runtime_in_event,
                       'kernel': get_kernel_time_in_event,
                       'memset': get_memset_time_in_event,
                       'memcpy': get_memcpy_time_in_event,
                       'sync': get_sync_time_in_event}.items():
            _times = [f(s, e) / 10**6 for s, e in event_start_end]  # ns -> ms
            mean = 0 if len(_times) < 1 else statistics.mean(_times)
            times[key].append(mean)
            stdev = 0 if len(_times) < 2 else statistics.stdev(_times)
            times[f'{key}_stdev'].append(stdev)

    df = pd.DataFrame(times, index=index)
    print(df)
    print(f'Writing results to "{pickle_path}"')
    df.to_pickle(pickle_path)


def get_kernel_timeline(main_event_text, sub_event_texts, pickle_path, main_event_indices=None):
    main_event_start_end = get_event_start_end(main_event_text)
    assert len(main_event_start_end) > 0, f'event {main_event_text} does not exist.'
    if main_event_indices is None:
        main_event_indices = [len(main_event_start_end) - 1]  # the last index
    main_event_indices = sorted(main_event_indices)
    print(main_event_start_end)
    target_start_end = [main_event_start_end[i] for i in main_event_indices]
    if main_event_text in sub_event_texts:
        sub_event_texts.remove(main_event_text)
    print(f'Collecting timeline for {sub_event_texts} in "{main_event_text}" event (indices:{main_event_indices})')
    timeline = {'start_end': target_start_end}
    print(timeline)
    for txt in sub_event_texts:
        event_start_end = get_event_start_end(txt)
        if len(event_start_end) == 0:
            continue
        timeline[txt] = []
        for s, e in event_start_end:
            if any(start <= s and e <= end for start, end in target_start_end):
                timeline[txt].append(get_kernel_start_end_in_event(s, e))
        print(txt, len(timeline[txt]))

    print(f'Writing results to "{pickle_path}"')
    with open(pickle_path, 'wb') as f:
        pickle.dump(timeline, f)


def main():
    if args.event_texts is None:
        event_texts = get_all_event_texts()
        event_texts = [t for t in event_texts if t is not None]
        if args.event_keywords is not None:
            event_keywords = args.event_keywords.split(',')
            event_texts = [txt for txt in event_texts
                           if any([kwd in txt for kwd in event_keywords])]
    else:
        event_texts = args.event_texts.split(',')
        if args.event_keywords is not None:
            warnings.warn('As event_texts is specified, event_keywords will be ignored.')

    if args.pickle_path_stats is not None:
        get_stats(event_texts, args.pickle_path_stats)

    if args.pickle_path_timeline is not None:
        if args.main_event_indices is not None:
            main_event_indices = [int(s) for s in args.main_event_indices.split(',')]
        else:
            main_event_indices = []
        get_kernel_timeline(args.main_event_text, event_texts, args.pickle_path_timeline, main_event_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite_path', type=str)
    parser.add_argument('--pickle_path_stats', type=str, default=None)
    parser.add_argument('--pickle_path_timeline', type=str, default=None)
    parser.add_argument('--ignore_first_event', action='store_true')
    parser.add_argument('--event_texts', type=str)
    parser.add_argument('--event_keywords', type=str)
    parser.add_argument('--main_event_indices', type=str, default=None)
    parser.add_argument('--main_event_text', type=str)
    args = parser.parse_args()
    con = sqlite3.connect(args.sqlite_path)
    main()
