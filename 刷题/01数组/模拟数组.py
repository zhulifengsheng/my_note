# 59. 螺旋矩阵
def f1(n):
    matrix = [[-1] * n for _ in range(n)]

    round_num = n // 2
    all_data = list(range(1, n**2+1))
    idx = 0

    for round_num_idx in range(round_num+1):
        start_i, start_j = round_num_idx, round_num_idx
        while idx < len(all_data):
            matrix[start_i][start_j] = all_data[idx]

            if start_i == round_num_idx and start_j < n-round_num_idx-1:
                start_j += 1

            elif start_j == n-round_num_idx-1 and start_i < n-round_num_idx-1:
                start_i += 1

            elif start_i == n-round_num_idx-1 and start_j > round_num_idx:
                start_j -= 1

            elif start_j == round_num_idx and start_i > round_num_idx:
                start_i -= 1
            
            idx += 1
            if start_i == round_num_idx and start_j == round_num_idx:
                break

    return matrix

if __name__ == "__main__":
    matrix = f1(3)

    print(matrix)
