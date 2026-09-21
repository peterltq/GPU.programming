## Game Description

This project implements a game where two GPU cores play **Tic-Tac-Toe** against each other.

There are two threads, **Thread A** and **Thread B**, which act as the two players. Each thread takes turns making a move.

The game proceeds as follows:

1. **Thread A** initiates the game by making a random move, then notifies **Thread B** to play.

2. **Thread B** makes a move, then notifies **Thread A** to continue.

3. After each move, the thread checks whether the game has a **winner** or has reached a **draw**.
   - If there is a winner or a draw, the result is printed and the game exits.
   - Otherwise, the threads continue alternating until one of the final game states is reached.

## GPU Computation

Each time it is **Thread A's** turn, it launches a GPU thread to compute and make its move.

Similarly, **Thread B** launches another GPU thread to compute and make its move.

## Thread Synchronization

The two CPU threads are synchronized using a **mutex** and **condition variable**.

Only one thread makes a move at a time. One thread waits until the other thread completes its move and signals it to continue.

Conceptually:

```text
Thread A
   │
   ├── Launch GPU kernel
   │
   ├── Make move
   │
   └── Notify Thread B
              │
              ▼
           Thread B
              │
              ├── Launch GPU kernel
              │
              ├── Make move
              │
              └── Notify Thread A
                         │
                         ▼
                       ...
```

## Unified Memory

The Tic-Tac-Toe board is stored in **CUDA Unified Memory**.

The board's memory pointer is accessible from both:

- **Host scope (CPU)**
- **Device scope (GPU)**

This allows both the CPU threads and GPU kernels to access the same game board.

## Demo

[View Demo](https://drive.google.com/drive/folders/0B2DPaR7njC2_ZW9VQ3lBRXVNVG8?resourcekey=0-57Agf4wtuoSR_h734S00hQ)

## Code Walkthrough

[View Code Walkthrough](https://drive.google.com/drive/folders/0B2DPaR7njC2_ZW9VQ3lBRXVNVG8?resourcekey=0-57Agf4wtuoSR_h734S00hQ)
