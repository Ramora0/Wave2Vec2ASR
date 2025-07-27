import matplotlib.pyplot as plt
import numpy as np

# Configuration
STRIDE = 50  # Configurable stride for compression ratio calculation
MAX_POSITION = 1500  # Maximum position to consider

boundaries = [[0, 52, 98, 100, 130, 170, 183, 190, 197, 199, 213, 219, 221, 225, 226, 234, 235, 237, 250, 255, 259, 261, 272, 282, 289, 300, 312, 319, 321, 323, 326, 330, 333, 340, 341, 342, 345, 348, 353, 356, 360, 372, 382, 389, 390, 401, 403, 409, 412, 413, 416, 418, 420, 424, 431, 432, 433, 438, 440, 448, 452, 454, 459, 462, 471, 472, 473, 474, 475, 480, 488, 490, 499, 503, 508, 514, 518, 520, 523, 525, 535, 536, 539, 542, 543, 550, 551, 553, 556, 558, 565, 569, 577, 581, 583, 587, 593, 596, 597, 598, 600, 603, 608, 615, 619, 634, 635, 640, 642, 644, 645, 646, 649, 653, 655, 664, 667, 674, 677, 679, 689, 691, 695, 697, 699, 700, 703, 707, 708, 711, 716, 720, 721, 728, 733, 739, 742, 744, 750, 753, 761, 762, 771, 772, 777, 785, 787, 791, 794, 797, 800, 803, 805, 806, 808, 809, 810, 811, 815, 820, 827, 829, 838, 842, 848, 849, 850, 852, 858, 863, 869, 875, 876, 877, 879, 889, 892, 898, 909, 915, 918, 919, 920, 922, 927, 929, 930, 938,
               940, 947, 950, 951, 952, 956, 958, 966, 975, 976, 978, 980, 982, 986, 1005, 1009, 1012, 1014, 1017, 1020, 1021, 1023, 1027, 1028, 1033, 1035, 1036, 1039, 1040, 1043, 1045, 1049, 1050, 1053, 1059, 1060, 1065, 1068, 1069, 1073, 1075, 1079, 1083, 1084, 1089, 1095, 1105, 1108, 1111, 1122, 1124, 1134, 1137, 1141, 1142, 1146, 1153, 1156, 1157, 1162, 1165, 1168, 1177, 1185, 1187, 1190, 1191, 1192, 1195, 1201, 1212, 1214, 1215, 1216, 1218, 1220, 1225, 1227, 1232, 1233, 1235, 1240, 1243, 1244, 1252, 1253, 1260, 1262, 1272, 1275, 1276, 1277, 1279, 1285, 1286, 1287, 1303, 1308, 1311, 1314, 1316, 1319, 1320, 1321, 1322, 1323, 1325, 1334, 1335, 1341, 1344, 1351, 1353, 1356, 1357, 1360, 1362, 1366, 1370, 1376, 1380, 1381, 1383, 1384, 1391, 1392, 1396, 1398, 1399, 1403, 1423, 1424, 1425, 1427, 1428, 1429, 1430, 1434, 1435, 1436, 1438, 1443, 1448, 1449, 1453, 1454, 1456, 1457, 1462, 1464, 1465, 1469, 1476, 1477, 1483, 1484, 1485, 1488, 1489, 1491]]

# Flatten the boundaries list (assuming it's a list of lists)
flat_boundaries = [boundary for sublist in boundaries for boundary in sublist]

# Filter boundaries to only include values from 0 to MAX_POSITION
filtered_boundaries = [b for b in flat_boundaries if 0 <= b <= MAX_POSITION]


def calculate_compression_ratios(boundaries, stride, max_position):
    """
    Calculate compression ratio at each point with given stride.
    Compression ratio = number of boundaries in window / window size
    """
    positions = []
    compression_ratios = []

    for pos in range(0, max_position, stride):
        window_end = min(pos + stride, max_position)
        window_size = window_end - pos

        # Count boundaries in this window
        boundaries_in_window = sum(
            1 for b in boundaries if pos <= b < window_end)

        # Calculate compression ratio (boundaries per unit length)
        compression_ratio = boundaries_in_window / window_size if window_size > 0 else 0

        positions.append(pos + stride/2)  # Use middle of window as x-position
        compression_ratios.append(compression_ratio)

    return positions, compression_ratios


# Calculate compression ratios
positions, compression_ratios = calculate_compression_ratios(
    filtered_boundaries, STRIDE, MAX_POSITION)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Original histogram
ax1.hist(filtered_boundaries, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Boundary Values')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Frequency Distribution of Boundaries (0-{MAX_POSITION})')
ax1.grid(True, alpha=0.3)

# Add some statistics to histogram
total_boundaries = len(filtered_boundaries)
mean_boundary = np.mean(filtered_boundaries)
median_boundary = np.median(filtered_boundaries)

ax1.axvline(mean_boundary, color='red', linestyle='--',
            label=f'Mean: {mean_boundary:.1f}')
ax1.axvline(median_boundary, color='green', linestyle='--',
            label=f'Median: {median_boundary:.1f}')
ax1.legend()

# Compression ratio plot
ax2.plot(positions, compression_ratios, 'b-',
         linewidth=2, marker='o', markersize=4)
ax2.set_xlabel('Position')
ax2.set_ylabel('Compression Ratio (boundaries per unit)')
ax2.set_title(f'Compression Ratio vs Position (Stride = {STRIDE})')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, MAX_POSITION)

# Add horizontal line for average compression ratio
avg_compression_ratio = np.mean(compression_ratios)
ax2.axhline(avg_compression_ratio, color='red', linestyle='--', alpha=0.7,
            label=f'Average: {avg_compression_ratio:.4f}')
ax2.legend()

plt.tight_layout()

# Print some statistics
print("="*60)
print("BOUNDARY STATISTICS")
print("="*60)
print(f"Configuration: Stride = {STRIDE}, Max Position = {MAX_POSITION}")
print(f"Total boundaries in range 0-{MAX_POSITION}: {total_boundaries}")
print(f"Mean boundary value: {mean_boundary:.2f}")
print(f"Median boundary value: {median_boundary:.2f}")
print(f"Min boundary value: {min(filtered_boundaries)}")
print(f"Max boundary value: {max(filtered_boundaries)}")

print("\n" + "="*60)
print("COMPRESSION RATIO STATISTICS")
print("="*60)
print(f"Average compression ratio: {avg_compression_ratio:.4f}")
print(f"Max compression ratio: {max(compression_ratios):.4f}")
print(f"Min compression ratio: {min(compression_ratios):.4f}")
print(f"Std dev of compression ratios: {np.std(compression_ratios):.4f}")

# Find windows with highest and lowest compression ratios
max_idx = compression_ratios.index(max(compression_ratios))
min_idx = compression_ratios.index(min(compression_ratios))
print(
    f"Highest compression at position {positions[max_idx]:.0f}: {compression_ratios[max_idx]:.4f}")
print(
    f"Lowest compression at position {positions[min_idx]:.0f}: {compression_ratios[min_idx]:.4f}")

plt.show()
