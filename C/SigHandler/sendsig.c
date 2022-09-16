///////////////////////////////////////////////////////////////////////////////
//
// Copyright 2020 Jim Skrentny
// Posting or sharing this file is prohibited, including any changes/additions.
// Used by permission, CS 354 Spring 2021, Deb Deppeler
//
////////////////////////////////////////////////////////////////////////////////
// Main File:        sendsig.c
// This File:        sendsig.c
// Other Files:      divison.c mySigHandler.c
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

#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/** This is the main fuction of this program
 * send signals to a process
 *
 * Makes sure user input is valid
 */
int main(int argc, char *argv[]){

    if (argc != 3) { 
        printf("Usage: <signal type> <pid>\n");
	exit(1);
    } else {
	int track = atoi(argv[2]);
	if (strcmp(argv[1],"-u") == 0){
            kill(track, SIGUSR1); 
        }
	if (strcmp(argv[1],"-i") == 0){
            kill(track, SIGINT);
	} 

    }
    return 0;
}
