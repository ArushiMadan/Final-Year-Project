/*******************************************************/
/*                                                     */
/*                     stop.c                          */
/*                     ======                          */
/*                                                     */
/*   C programme to remove stoplist words from texts   */
/*                                                     */
/*        Version 1: Martin Russell                    */
/*                                                     */
/* Dept. Electronic, Electrical & Computer Engineering */
/*            University of Birmingham                 */
/*                                                     */
/*    To compile under linux:                          */
/*                 gcc stop.c  %                        */
/*                 mv a.out stop                       */
/*                                                     */
/*    To run:                                          */
/*                 stop stoplist text                  */
/*                                                     */
/*******************************************************/

#define MAX_STR_LEN 512
#define MAX_LIN_LEN 30

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>


/********************************************************/

/* Function to convert upper case letter in a token to lower case */
void upper2lower(char *token)
{
  int c;
  for (c=0; c<strlen(token); c++) token[c]=tolower(token[c]);
}

/********************************************************/

/* Function to remove punctuation from start and end of word */
int removePunct(char *token)
{
  int l;
  int r; /* counts number of characters removed */
  r=0;
  l=strlen(token);
  /* first remove punctuation from the end of the word */
  /* 'isalnum[x]' returns true if x is a letter or number */
  while ((!isalnum(token[l-1])) && (strlen(token)>0))
    {
      token[l-1]='\0';
      l--;
      r++;
    }
  /* now remove punctuation from the start of the word */
  while ((!isalnum(token[0])) && (strlen(token)>0))
    {
      int c;
      l=strlen(token);
      for (c=0; c<l; c++) token[c]=token[c+1];
      token[c]='\0';
      r++;
    }
  return(r);
}

/********************************************************/

/* Function to read next word from text */
int nextWord(FILE *ip, char *token)
{
  int x;
  int c;
  for (c=0; c<MAX_STR_LEN; c++) token[c]='\0';
  x=fscanf(ip,"%s",token);
  if (x != EOF)
    {
      upper2lower(token);
      removePunct(token);
    }
  return x;
}

/********************************************************/

/* Function to read stop lists into a char array */

int readStopList(FILE *stopFile, char ***stopList)
{
  int w;
  char wrd[MAX_STR_LEN];
  /* count words in stop list */
  w=0;
  while (nextWord(stopFile,wrd)!=EOF) w++;
  /* create memory for stop list */
  *stopList=calloc(w,sizeof(char*));
  /* go back to start of stop list */
  rewind(stopFile);
  w=0;
  while (nextWord(stopFile,wrd)!=EOF)
    {
      int c;
      /* increment stop word counter */
      /* allocate memory for string */
      (*stopList)[w]=calloc(strlen(wrd),sizeof(char));
      for (c=0; c<strlen(wrd); c++) (*stopList)[w][c]=wrd[c];
      (*stopList)[w][c]='\0';
      w++;
    }
  return w;
}

/********************************************************/

/* Function to remove stop words from text */
int removeStopWords(FILE *textFile, char **stopList, int numStopWds)
{
  int s;
  int line;
  char wrd[MAX_STR_LEN];

  /* initialise counters */
  s=0;
  line=0;
  /* go through text file */
  while (nextWord(textFile,wrd)!=EOF)
    {
      int i;
      int stopWd;

      /* initialise flag to indicate stop word */
      stopWd=0;
      for (i=0; i<numStopWds; i++)
	{
	  if (strcmp(stopList[i],wrd)==0)
	    {
	      stopWd=1;
	      s++;
	    }
	}
      if (stopWd==0)
	{
	  printf("%s ",wrd);
	  line+=strlen(wrd);
	  if (line>MAX_LIN_LEN)
	    {
	      printf("\n");
	      line=0;
	    }
	}
    }
  return s;
}

int main(int argc, char *argv[])
{
  FILE *stopFile;
  FILE *txtFile;

  char **stopList;

  int numStopWds;
  int totSW;
  int w;

  /* Check correct number of input parameters */
  if (argc!=3)
    {
      printf("format: stop stopListFile textFileName\n");
      exit(1);
    }

  /* set pointer to input stop list file */
  argv++;
  if ((stopFile=fopen(*argv,"r"))==NULL)
    {
      printf("Error: can't open stopList file %s\n",*argv);
      exit(1);
    }

  /* set pointer to input text file */
  argv++;
  if ((txtFile=fopen(*argv,"r"))==NULL)
    {
      printf("Error: can't open text file %s\n",*argv);
      exit(1);
    }

  /* read in stop list */
  numStopWds=readStopList(stopFile,&stopList);

  /* printf("num stopWords = %d\n",numStopWds); */
  /* for (w=0;w<numStopWds;w++) */
  /*  { */
  /*    printf("w=%d, %s\n",w,stopList[w]); */
  /*  } */

  /* initialise stop word counter */
  totSW=0;
  /* remove stop words from text file */
  totSW+=removeStopWords(txtFile,stopList,numStopWds);
  return(totSW);
}
