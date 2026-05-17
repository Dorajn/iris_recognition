MATCH_THRESHOLD = 0.41
SHIFT_RANGE = 48
SHIFT_COARSE_STEP = 4

# Normalizacja paska tęczówki
NORMALIZED_HEIGHT = 64
NORMALIZED_WIDTH = 512

# Stabilizacja segmentacji: wspólny środek źrenicy/tęczówki (0 = osobne centra)
IRIS_CENTER_BLEND = 0.75

# Filtry Gabora (4 orientacje × 2 fazy = 8 bitów na piksel)
GABOR_KERNEL_SIZE = 21
GABOR_SIGMA = 2.5
GABOR_LAMBDA = 6.0
GABOR_GAMMA = 0.5
GABOR_ORIENTATIONS = (0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)
