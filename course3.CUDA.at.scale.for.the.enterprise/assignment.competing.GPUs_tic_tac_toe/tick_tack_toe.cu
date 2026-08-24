#include <stdio.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <stdlib.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <tuple>

#include "tick_tack_toe.h"


pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond1, cond2;
volatile int gNextToMove = 0;

__host__ __device__ void printBoard(int *board) {
    printf("board:[\n");
    for (int i = 0; i<9; ++i) {
        printf("%d,", board[i]);
        if ((i+1) % 3 == 0) {
            printf("\n");
        }
    }
    printf("]\n");
}

__global__ void play(int *board, int idx, int *winner) {
    int rowStats[3][3]; 
    int colStats[3][3]; 
    int diagStats[2][3]; 

    // Scan the board and collect statistics for row/col/diag.
    for (int i = 0; i<3; ++i) {
        // check all rows. 
        auto[emptyCount, myCount, otherCount] = checkRow(idx, board, i);
        printf("gpu:%d, checkRow: return emptyCount:%d, myCount:%d, otherCountLeft:%d for row:%d\n", idx, emptyCount, myCount, otherCount, i);
        rowStats[i][0] = emptyCount;
        rowStats[i][1] = myCount;
        rowStats[i][2] = otherCount;
    }

    for (int j = 0; j<3; ++j) {
        // check all columns. 
        auto[emptyCount, myCount, otherCount] = checkColumn(idx, board, j);
        printf("gpu:%d, checkColumn: return emptyCount:%d, myCount:%d, otherCount:%d for col:%d\n", idx, emptyCount, myCount, otherCount, j);
        colStats[j][0] = emptyCount;
        colStats[j][1] = myCount;
        colStats[j][2] = otherCount;
    }

    auto[emptyCountLeft, myCountLeft, otherCountLeft] = checkLeftDiagonal(idx, board, 1, 1);
    printf("gpu:%d, check Left diag: return emptyCount:%d, myCount:%d, otherCount:%d\n", idx, emptyCountLeft, myCountLeft, otherCountLeft);
    diagStats[0][0] = emptyCountLeft;
    diagStats[0][1] = myCountLeft;
    diagStats[0][2] = otherCountLeft;
    
    auto[emptyCountRight, myCountRight, otherCountRight] = checkRightDiagonal(idx, board, 1, 1);
    printf("gpu:%d, check Right diag: return emptyCount:%d, myCount:%d, otherCount:%d\n", idx, emptyCountRight, myCountRight, otherCountRight);
    diagStats[1][0] = emptyCountRight;
    diagStats[1][1] = myCountRight;
    diagStats[1][2] = otherCountRight;

    // Calcualte and return the best move.
    auto[optimizedType, chosenRowIdx, chosenColIdx, isLost] = calculateMove(idx, rowStats, colStats, diagStats);

    if (isLost) {
        int tmpWinner = (idx == 1)? 2 : 1;
        *winner = tmpWinner;
        printf("gpu:%d: WINNER is %d\n", idx, tmpWinner);
        return;
    }

    printf("gpu:%d, optimizedType:%d, chosenRowIdx:%d, chosenColIdx:%d\n", idx, optimizedType, chosenRowIdx, chosenColIdx);
    
    // Now know types to play at, then 
    switch (optimizedType) {
        case 1:
            setCellInRow(idx, board, chosenRowIdx);
             break;
        case 2:
            setCellInColumn(idx, board, chosenColIdx);
            break;
        case 3:
            setCellInLeftDiagnonal(idx, board, 1, 1);
            break;
        case 4:
            setCellInRightDiagnonal(idx, board, 1, 1);
            break;
        default:
            break;
    }
    
    printBoard(board);

    return;

}

// Return a tuple: <optimizedType, chosenRow, chosenCol, doILose>
__device__ std::tuple<int, int, int, bool> calculateMove(int idx, int rowStats[][3], int colStats[][3], int diagStats[][3]) {
    ///// Step-1: Check if the other player already won.
    // Row. 
    for (int x = 0; x < 3; ++x) {
        if (rowStats[x][2] == 3) {
            // I lost.
            return {-1, -1, -1, true};
        }
    }
    // Col.
    for (int x = 0; x < 3; ++x) {
        if (colStats[x][2] == 3) {
            // I lost.
            return {-1, -1, -1, true};
        }
    }
    // Left Diag.
    if (diagStats[0][2] == 3) {
        return {-1, -1, -1, true};
    } 

    // Right Diag.
    if (diagStats[1][2] == 3) {
        return {-1, -1, -1, true};
    } 
    
    ////// Step-2: Firstly check if I already have 2 the same row/col/diag.
    // Row.
    for (int x = 0; x < 3; ++x) {
        int emptyCount = rowStats[x][0];
        int myCount = rowStats[x][1];
        if (emptyCount > 0 && myCount == 2) {
            // Need to fill the 3rd cell in this move. So return directly.
            return {1, x, -1, false};
        }
    }
    // Column.
    for (int x = 0; x < 3; ++x) {
        int emptyCount = colStats[x][0];
        int myCount = colStats[x][1];
        if (emptyCount > 0 && myCount == 2) {
            return {2, -1, x, false};
        }
    }

    // Left Diag.
    if (diagStats[0][0] > 0 && diagStats[0][1] == 2) {
        return {3, -1, -1, false};
    }

    // Right Diag.
    if (diagStats[1][0] > 0 && diagStats[1][1] == 2) {
        return {4, -1, -1, false};
    }

    ////// Step-3: Firstly check if any other move has 2 in the same row/col/diag.
    // Row.
    for (int x = 0; x < 3; ++x) {
        int emptyCount = rowStats[x][0];
        int otherCount = rowStats[x][2];
        if (emptyCount > 0 && otherCount == 2) {
            // Need to fill the 3rd cell in this move. So return directly.
            return {1, x, -1, false};
        }
    }
    // Column.
    for (int x = 0; x < 3; ++x) {
        int emptyCount = colStats[x][0];
        int otherCount = colStats[x][2];
        if (emptyCount > 0 && colStats[x][2] == 2) {
            return {2, -1, x, false};
        }
    }

    // Left Diag.
    if (diagStats[0][0] > 0 && diagStats[0][2] == 2) {
        return {3, -1, -1, false};
    }

    // Right Diag.
    if (diagStats[1][0] > 0 && diagStats[1][2] == 2) {
        return {4, -1, -1, false};
    }
    
    //// Step-4: I am safe now. Check myCount and get the most one in row/col/diag.
    int factor = -1;
    int maxChanceFactor = -1;
    // Optimization Type to go towards: 
    // 1:row, 2:col, 3:LeftDiag, 4: rightDiag;
    int optimizedType = -1; 
    int maxRowIdx = -1;
    int maxColIdx = -1;
    // Row.
    for (int i = 0; i<3; ++i) {
        int emptyCount = rowStats[i][0];
        int myCount = rowStats[i][1];
        int otherCount = rowStats[i][2];
        int chanceFactory = getChanceFactor(myCount, otherCount);
        if (emptyCount > 0 &&  chanceFactory > maxChanceFactor) {
            maxChanceFactor = chanceFactory;
            maxRowIdx = i;
            optimizedType = 1;
        }
    }
    // Column.
    for (int i = 0; i<3; ++i) {
        int emptyCount = colStats[i][0];
        int myCount = colStats[i][1];
        int otherCount = colStats[i][2];
        int chanceFactory = getChanceFactor(myCount, otherCount);
        if (emptyCount > 0 && chanceFactory > maxChanceFactor) {
            maxChanceFactor = chanceFactory;
            maxColIdx = i;
            optimizedType = 2;
        }
    }

    //Left Diag.
    int emptyCountLeft = diagStats[0][0];
    int myCountLeft = diagStats[0][1];
    int otherCountLeft = diagStats[0][2];
    int chanceFactoryLeft = getChanceFactor(myCountLeft, otherCountLeft);
    if (emptyCountLeft > 0 && chanceFactoryLeft > maxChanceFactor) {
        maxChanceFactor = chanceFactoryLeft;
        optimizedType = 3; //Left direction.
    }

    // Right Diag.
    int emptyCountRight = diagStats[1][0];
    int myCountRight = diagStats[1][1];
    int otherCountRight = diagStats[1][2];
    int chanceFactoryRight = getChanceFactor(myCountRight, otherCountRight);
    if (emptyCountRight > 0 && chanceFactoryRight > maxChanceFactor) {
        maxChanceFactor = chanceFactoryRight;
        optimizedType = 4; // Right direction.
    }
    printf("gpu:%d, maxChanceFactor:%d\n", idx, maxChanceFactor);
    return {optimizedType, maxRowIdx, maxColIdx, false};
}

// Calcualte a chance factor whose value favors more myCount and less otherCount.
__device__ int getChanceFactor(int myCount, int otherCount) {
    return (1 + myCount) * (5 - otherCount);
}

__host__ int genRandom() {
    // Seed the random number generator with current time
    srand(static_cast<unsigned int>(time(nullptr)));   
    return rand() % 9;
} 

// Randomly choose one cell to start.
__host__ int firstMove(int idx, int *board) {
    int randomNum = genRandom();
    int i = randomNum / 3;
    int j = randomNum % 3;

    board[3*i+j] = idx;
    return 0;
}

// return emptycount,myCount on the same row.
__device__ std::tuple<int, int, int> checkRow(int idx, int *board, int i) {
    int emptyCount = 0, myCount = 0, otherCount=0;
    for (int j=0; j<3; ++j) {
        if (board[3*i+j] == idx) {
            myCount++;
        } else if (board[3*i+j] == 0) {
            emptyCount++;
        } else {
            otherCount++;
        }
    }
    return {emptyCount, myCount, otherCount};
}

// return emptycount,myCount on the same column.
__device__ std::tuple<int, int, int> checkColumn(int idx, int *board, int j) {
    int emptyCount = 0, myCount = 0, otherCount=0;
    for (int i=0; i<3; ++i) {
        if (board[3*i+j] == idx) {
            myCount++;
        } else if (board[3*i+j] ==0) {
            emptyCount++;
        } else {
            otherCount++;
        }
    }
    return {emptyCount, myCount, otherCount};
}

// return emptycount,myCount on the diagonal.
__device__ std::tuple<int, int, int> checkLeftDiagonal(int idx, int *board, int i, int j) {
    int emptyCount = 0, myCount = 0, otherCount = 0;

    // diagonal of /;
    if (i + j == 2) {
        for (int ii = 0 ; ii < 3 ; ++ii) {
            int jj = 2 - ii;
            if (board[3*ii+jj] == 0) {
                emptyCount++;
            } else if (board[3*ii+jj] == idx) {
                myCount++;
            } else {
                otherCount++;
            }
        }
    }
    return {emptyCount, myCount, otherCount};
} 
    
// return emptycount,myCount on the diagonal.
__device__ std::tuple<int, int, int> checkRightDiagonal(int idx, int *board, int i, int j) {
    int emptyCount = 0, myCount = 0, otherCount = 0;

    for (int iii = 0 ; iii < 3 ; ++iii) {
        int jjj = iii;
        if (board[3*iii+jjj] == 0) {
            emptyCount++;
        } else if (board[3*iii+jjj] == idx) {
            myCount++;
        } else {
            otherCount++;
        } 
    }
    return {emptyCount, myCount, otherCount};
} 

__device__ int setCellInRow(int idx, int *board, int i) {
    for (int j=0; j<3; ++j) {
        if (board[3*i+j] == 0) {
            board[3*i+j] = idx;
            return j;
        }
    }
}

__device__ int setCellInColumn(int idx, int *board, int j) {
    for (int i=0; i<3; ++i) {
        if (board[3*i+j] == 0) {
            board[3*i+j] = idx;
            return i;
        }
    }
}

__device__ int setCellInRightDiagnonal(int idx, int *board, int i, int j) {
    for (int k=0; k<3; ++k) {
        int l = k;
        if (board[3*k+l] == 0) {
            board[3*k+l] = idx;
            return i;
        }
    }
    return 0;
}

__device__ int setCellInLeftDiagnonal(int idx, int *board, int i, int j) {
    // diag /
    for (int k=0; k<3; ++k) {
        int l = 2 - k;
        if (board[3*k+l] == 0) {
            board[3*k+l] = idx;
            return i;
        }
    }
}

__host__ int getEmptyCellCount(int *board) {
    int count = 0;
    for (int i = 0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            if (board[3*i+j] == 0) {
                count++;
            }
        }
    }
    return count;
}

struct ThreadData {
    int* board;
    int* winner;
};

void * cudaThreadFunction1(void* arg) {
    printf("Thread:1, Start.\n");
    ThreadData* data = (ThreadData*)arg;

    int* board = data->board;
    int* winner = data->winner;
    int idx = 1;

    // 1st threads initialize the board and make the 1st move.
    for (int i = 0; i<9; ++i) {
        board[i] = 0;
    }
    printBoard(board);
    printf("Thread:1, make the 1st move by thread-1.\n");
    firstMove(1, board);
    printf("Thread:1, after 1st move by thread-1.\n");
    printBoard(board);
    
    gNextToMove = 2;
    // Sleep for thread-2 to be in waiting mode.
    sleep(5);
    printf("Thread:1, After sleep in thread-1.\n");
    printf("Thread:1, Notif thread-2 to start.\n");
    pthread_cond_broadcast(&cond1);

    while (*winner == 0 && getEmptyCellCount(board) > 0) {
        pthread_mutex_lock(&gMutex);
        printf("Thread:1, Waiting for thread-2 to complete.\n");
        while (gNextToMove != 1) {
            pthread_cond_wait(&cond2, &gMutex);
        }
        printf("Thread:1, thread-2 is done, now it's thread-1 move\n");

        play<<<1, 1>>>(board, idx, winner);
        
        sleep(1);
        printf("Thread:1, broadcast that thread-1 completed the move.\n");
        gNextToMove = 2;
        pthread_cond_broadcast(&cond1);
        pthread_mutex_unlock(&gMutex);
    }
    if (*winner == 0) {
        printf("Thread:%d, DRAW\n", idx);
    }
    printf("Thread:1, End of thread-1.\n");
}

void * cudaThreadFunction2(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    int* board = data->board;
    int* winner = data->winner;
    int idx = 2;

    // 2st threads initialize the board and make the 1st move.
    printf("Thread:2, Start.\n");

    while (*winner == 0 && getEmptyCellCount(board) > 0) {
        pthread_mutex_lock(&gMutex);
        printf("thread:2, Wait till thread-1 is done.\n");
        while (gNextToMove != 2) {
            pthread_cond_wait(&cond1, &gMutex);
        }
        printf("Thread:2, Starting move\n");

        play<<<1, 1>>>(board, idx, winner);
        printf("Thread:2, completed the move, notif thread-1\n");
        
        sleep(1);

        gNextToMove = 1;
        pthread_cond_broadcast(&cond2);
        pthread_mutex_unlock(&gMutex);
    }
    if (*winner == 0) {
        printf("thread:%d, DRAW\n", idx);
    }
    printf("thread:2, End of thread-2.\n");
}


int main() {
    printf("Start\n");

    pthread_cond_init(&cond1, nullptr);
    pthread_cond_init(&cond2, nullptr);

    int *globalBoard;
    int *winner;
    cudaMallocManaged(&globalBoard, 9 * sizeof(int));
    cudaMallocManaged(&winner, sizeof(int));
    *winner = 0; // no one.

    pthread_t thread1, thread2;
    ThreadData threadData1, threadData2;

    threadData1.board = globalBoard;
    threadData1.winner = winner;
    threadData2.board = globalBoard;
    threadData2.winner = winner;

    printf("Creating thread #1.\n");
    pthread_create(&thread1, nullptr, cudaThreadFunction1, &threadData1);

    printf("Creating thread #2.\n");
    pthread_create(&thread2, nullptr, cudaThreadFunction2, &threadData2);

    void* status1, *status2;
    pthread_join(thread1, &status1);
    pthread_join(thread2, &status2);
    printf("Both threads are completed.\n");

    cudaFree(globalBoard);
    cudaFree(winner);

    printf("All done\n");
    return 0;
}