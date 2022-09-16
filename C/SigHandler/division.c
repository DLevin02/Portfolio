///////////////////////////////////////////////////////////////////////////////
//
// Copyright 2020 Jim Skrentny
// Posting or sharing this file is prohibited, including any changes/additions.
// Used by permission, CS 354 Spring 2021, Deb Deppeler
//
////////////////////////////////////////////////////////////////////////////////
// Main File:        division.c
// This File:        division.c
// Other Files:      sendsig.c mySigHandler.c
// Semester:         CS 354 Fall 2021
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
//
// Copyright 2013,2019-2020
// Posting or sharing this file is prohibited, including any changes/additions.
//
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <time.h>

int positives = 0;

/**
 * handle_SIGFPE:
 * signal handler function to catch when division by 0
 * is attempted and exit program after displaying final
 * statistics
 */
void handle_SIGFPE(){
    printf("Error: a division by 0 operation was attempted.\n");
    printf("Total number of operations completed successfully: %i\n", positives);
    printf("The program will be terminated.\n");
    exit(0);
}

/**
 * handle_SIGINT:
 * signal handler function to catch when the
 * keyboard interrupt signal (^C) is entered
 * and exit program after displaying final statistics
 */
void handle_SIGINT(){
    printf("\nTotal number of operations completed successfully: %i\n", positives);
    printf("The program will be terminated.\n");
    exit(0);
}

/**
 * main:
 * registers both signal handlers, runs an infinite loop
 * that asks user for two integers and displays quotient
 * with remainder, and calls signal handler functions when
 * needed
 */
int main(){
    
    struct sigaction sigfpe;
    memset(&sigfpe, 0, sizeof(sigfpe));
    sigfpe.sa_handler = &handle_SIGFPE;
    if(sigaction(SIGFPE, &sigfpe, NULL) != 0){
        printf("ERROR binding SIGFPE handler\n");
        exit(1);
    }

    struct sigaction sigint;
    memset(&sigint, 0, sizeof(sigint));
    sigint.sa_handler = &handle_SIGINT;
    if(sigaction(SIGINT, &sigint, NULL) != 0){
        printf("ERROR binding SIGINT handler\n");
        exit(1);
    }
	
    while(1){
        int f;
        int s;
        char buffer[100];
        printf("Enter first integer: ");
        fgets(buffer, 100, stdin);
        f = atoi(buffer);
        printf("Enter second integer: ");
        fgets(buffer, 100, stdin);
        s = atoi(buffer);

        int q = f / s;
        int r = f % s;

        printf("%i / %i is %i with a remainder of %i\n", f, s, q, r);
        
	positives++;
    }
    
}


