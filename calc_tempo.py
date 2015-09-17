__author__ = 'shawn'


def get_mean_tempo(file_name):
    lines = [line.rstrip('\n') for line in open(file_name)]
    buckets = {}
    for line in lines:
        sum_diff = 0
        total = 0
        points = line.split('\t')
        for i in range(1,len(points)):
            sum_diff += float(points[i]) - float(points[i-1])
            total += 1
        avg = sum_diff/total
        if str(avg)[:3] in buckets:
            buckets[str(avg)[:3]] += [avg]
        else:
            buckets[str(avg)[:3]] = [avg]
        print(avg)
    highest_bucket = find_largest_bucket(buckets)
    total_sum_diff = 0
    total_total = 0
    for point in buckets[highest_bucket]:
        total_sum_diff += point
        total_total += 1
    print(total_sum_diff/total_total)
    print(60/(total_sum_diff/total_total))
    print()
    print(buckets)

def find_largest_bucket(bucket):
    highest = 0
    highest_key = ''
    for key, value in bucket.items():
        if len(value) > highest:
            highest = len(value)
            highest_key = key
    return highest_key

get_mean_tempo("train1.txt")
