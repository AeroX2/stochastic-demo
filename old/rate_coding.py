class RateCoding:

    def __init__(self, time_period):
        self.time_period = float(time_period)

    def spikes(self, stream):
        count = 0
        count_bit = 0
        while stream > 0 or count != 0:
            count_bit += 1 if stream&1 else 0

            count += 1
            if (count >= self.time_period):
                yield count_bit/self.time_period
                count = 0
                count_bit = 0
            stream >>= 1
    
r = RateCoding(8)
z =  '00010101' # 3/8 = 0.375
z += '10011010' # 4/8 = 0.5
z += '01101111' # 6/8 = 0.75
z += '11111100' # 6/8 = 0.75
z += '00001000' # 1/8 = 0.125
z = z[::-1]

z = int(z, 2)
print(list(r.spikes(z)))
