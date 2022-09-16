///////////////////////////////////////////////////////////////////////////////
//
// Copyright 2020 Jim Skrentny
// Posting or sharing this file is prohibited, including any changes/additions.
// Used by permission, CS 354 Spring 2021, Deb Deppeler
//
////////////////////////////////////////////////////////////////////////////////
// Main File:        mySigHandler.c
// This File:        mySigHandler.c
// Other Files:      division.c sendsig.c
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
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>

int track = 0; 

/**
 * Used to handle the SIGALRM singal
 * Print PID, current time for every 3 seconds
 */
void handle_SIGALRM(){
    time_t cur;
    time(&cur);
    if (cur == -1 || ctime(&cur) == NULL) {
        printf("Error Calculating Time.\n");
        exit(0);
    }
    printf("PID: %d CURRENT TIME: %s", getpid(), ctime(&current));
    alarm(3);
}

/**
 * handle_SIGUSR1:
 * signal handler function for SIGUSR1 that
 * counts number of times SIGUSR1 is caught
 */
void handle_SIGUSR1(){
    track++;
    printf("SIGUSR1 has been handled!\n");
}

/**
 * Used to handle the SIGINT singal
 * Print the total number of SIFUSR1 caught and exit the program
 */
void handle_SIGINT(){
    printf("\nSIGINT received.\n");
    printf("SIGUSR1 was received %d times. Exiting now.\n", track);
    exit(0);
}

/**
 * main:
 * sets an alarm, registers signal handlers, and runs
 * an infinite loop
 */
int main(){
    printf("Pid and time print every 3 seconds.\n");
    printf("Enter Ctrl-C to end the program.\n");

    alarm(3);
    
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_handler = &handle_SIGALRM;
    if(sigaction(SIGALRM, &act, NULL) != 0){
        printf("ERROR binding SIGALRM handler\n");
	exit(1);
    }
    
    struct sigaction act2;
    memset(&act2, 0, sizeof(act2));
    act2.sa_handler = &handle_SIGUSR1;
    if(sigaction(SIGUSR1, &act2, NULL) != 0){
        printf("ERROR binding SIGUSR1 handler\n");
        exit(1);
    }

    struct sigaction act3;
    memset(&act3, 0, sizeof(act3));
    act3.sa_handler = &handle_SIGINT;
    if(sigaction(SIGINT, &act3, NULL) != 0){
        printf("ERROR binding SIGINT handler\n");
        exit(1);
    }
	
    while(1){
    }
}

