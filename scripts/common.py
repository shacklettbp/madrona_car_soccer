def print_elos(elos):
    chunk_size = 4
    num_chunks = -(elos.shape[0] // -chunk_size)

    for i in range(num_chunks):
        for j in range(chunk_size):
            idx = i * chunk_size + j
            if idx >= elos.shape[0]:
                break;

            elo = elos[idx]

            print(f"{idx:>3}: {elo:>8.2f}       ", end='')

        print()
