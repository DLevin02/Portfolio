///////////////////////////////////////////////////////////////////////////////
//
// Copyright 2021 Deb Deppeler
// Posting or sharing this file is prohibited, including any changes/additions.
//
// We have provided comments and structure for this program to help you get 
// started.  Later programs will not provide the same level of commenting,
// rather you will be expected to add same level of comments to your work.
// 09/20/2021 Revised to free memory allocated in get_board_size function.
//
////////////////////////////////////////////////////////////////////////////////
// Main File:        myMagicSquare.c
// This File:        myMagicSquare.c
// Other Files:      (name of all other files if any)
// Semester:         CS 354 Fall 2021
// Instructor:       deppeler
//
// Author:           Drew Levin
// Email:            dslevin2@wisc.edu
// CS Login:         dlevin
//
/////////////////////////// OTHER SOURCES OF HELP //////////////////////////////
//                   Fully acknowledge and credit all sources of help,
//                   other than Instructors and TAs.
//
// Persons:          Identify persons by name, relationship to you, and email.
//                   Describe in detail the the ideas and help they provided.
//
// Online sources:   Avoid web searches to solve your problems, but if you do
//                   search, be sure to include Web URLs and description of
//                   of any information you find.
////////////////////////////////////////////////////////////////////////////////
   

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure that represents a magic square
typedef struct {
    int size;           // dimension of the square
    int **magic_square; // pointer to heap allocated magic square
} MagicSquare;

/* TODO:
 * Prompts the user for the magic square's size, reads it,
 * checks if it's an odd number >= 3 (if not display the required
 * error message and exit), and returns the valid number.
 */
int getSize() {
	int size;

	// input the size of the magic square
	printf("Enter the size: ");
	scanf("%d",&size);

	// Check The Size of The Magic Square
	if(size < 3 || size%2 == 0){
		printf("ERROR: Size for Magic Square must be ODD and greater than or equal to 3.");
		exit(1);
	}
	return size;
}
   
/* TODO:
 * Makes a magic square of size n using the alternate 
 * Siamese magic square algorithm from assignment and 
 * returns a pointer to the completed MagicSquare struct.
 *
 * n the number of rows and columns
 */
MagicSquare *generateMagicSquare(int n) {
    // allocate memory
    int **board = malloc(sizeof(int*) * n);
    if(board == NULL) {
        printf("Could Not Allocate Memory");
        exit(1);
    }
    for(int i = 0; i < n; i++) {
        *(board + i) = malloc(sizeof(int) * n);
        if((board + 1) == NULL) {
            printf("Could Not Allocate Memory");
            exit(1);
        }
    }
    // Set ALl Array Elements to 0
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            *(*(board + i) + j) = 0;
        }
    }
    
    int i = 0;
    int j = n/2;
    
    *(*(board + i) + j) = 1;

    for(int tracker = 2; tracker <= n*n; tracker++) {
        if(i == 0 && j == (n-1)) {
            if(*(*(board + (n-1)) + 0) != 0) {
                i = i + 1;
                *(*(board + i) + j) = tracker;
                continue;
            }

            i = n-1;
            j = 0;
            *(*(board + i) + j) = tracker;

        }else if(i == 0 && j != (n - 1)) {
            if(*(*(board + (n - 1)) + (j + 1)) != 0) {
                i = i + 1;
                *(*(board + i) + j) = tracker;
                continue;
            }

	    j = j+1;
            i = n - 1;
            *(*(board + i) + j) = tracker;

        }else if(i != 0 && j == (n-1)) {
            if(*(*(board + (i-1)) + 0) != 0) {
                if(i != (n-1)) {
                    i = i + 1;
                    *(*(board + i) + j) = tracker;
                    continue;
                }else{
                    i = 0;
                    *(*(board + i) + j) = tracker;
                    continue;
                }
            }

            i = i - 1;
            j = 0;
            *(*(board + i) + j) = tracker;

        }else{
            if(*(*(board + (i-1)) + (j+1)) != 0) {
                i = i+1;
                *(*(board + i) + j) = tracker;
                continue;
            }

            i = i-1;
            j = j+1;
            *(*(board + i) + j) = tracker;

        }
    }

    // Setup Struct
    MagicSquare *magic_pointer =  malloc(sizeof(MagicSquare));

    magic_pointer -> magic_square = board;

    magic_pointer -> size = n;

    board = NULL;

    // return the struct
    return magic_pointer;

} 

/* TODO:  
 * Opens a new file (or overwrites the existing file)
 * and writes the square in the specified format.
 *
 * magic_square the magic square to write to a file
 * filename the name of the output file
 */
void fileOutputMagicSquare(MagicSquare *magic_square, char *filename) {
    int board_size = magic_square->size;

    int **magic_board = magic_square->magic_square;
    // open file
    FILE *fp = fopen(filename, "w");

    if(fp == NULL) {
        printf("Could Not Open The File.");
        exit(1);
    }

    fprintf(fp, "%i",board_size);
    fputc('\n', fp);

    for(int j = 0; j < board_size; j++) {
        for(int i = 0; i < board_size; i++) {
            fprintf(fp, "%i", *(*(magic_board + j) + i));
            if (i == (board_size - 1)){
		    break;
	    }
            fputc(',', fp);
        }
        fputc('\n', fp);
    }
    // close file
    fclose(fp);
}

/* TODO:
 * Generates a magic square of the user specified size and
 * output the quare to the output filename
 */
int main(int argc, char *argv[]) {
    // TODO: Check input arguments to get output filename
	if(argc != 2){
		printf("Not the Correct Num of Args");
		exit(1);
	}
	char *fileName = *(argv+1);
	printf("Enter a size for the Board that is odd and greater than or equal to 3");
	int size;
	scanf("%i", &size);

	if((size < 3) | (size%2 == 0)){
		printf("Not The Correct Input");
		exit(1);
	}
    // TODO: Generate the magic square
	MagicSquare *matrix_pointer = generateMagicSquare(size);
    // TODO: Output the magic square
    	fileOutputMagicSquare(matrix_pointer, fileName);

	int **free_pointer = matrix_pointer->magic_square;

	size = matrix_pointer->size;

	for(int j = 0; j < size; j++){
		free(*(free_pointer + j));
	}
	free(free_pointer);
	free_pointer = NULL;
	
	free(matrix_pointer);

    return 0;
} 

//  F21 deppeler myMagicSquare.c      

