/*******************************************************/
/*                                                     */
/*  C programme to create term - document index file   */
/*                                                     */
/*        Version 1: Martin Russell 	               */
/*                                                     */
/* Dept. Electronic, Electrical & Computer Engineering */
/*            University of Birmingham                 */
/*                                                     */
/*    To compile under linux:                          */
/*                 gcc -lm index.c                     */
/*                 mv a.out index                      */
/*                                                     */
/*    To run:                                          */
/*                 index scriptFile                    */
/*                                                     */
/*******************************************************/

#define MAX_STR_LEN 512

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/********************************************************/

/* structure to store document lengths */
struct docData {
  char *docName;
  float length;
};

/********************************************************/

/* structure to store document index item */
struct docIndex {
  char *docName;
  int count;
  float weight;
  struct docIndex *nextIndex;
};

/********************************************************/

/* struture to store linked list of words */
struct item {
  char *text;
  int wrdCount;
  int docCount;
  float IDF;
  struct docIndex *docList;
  struct item *nextItem;
};

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

/* Function to check linked list */
int checkList(struct item root)
{
  int i;
  int d;
  struct item *curItem;
  struct docIndex *curDoc;

  /* initialise */
  i=1;
  curItem=root.nextItem;
  curDoc=curItem->docList;
  printf("%d %d\n",root.wrdCount,root.docCount);
  while ((curItem->nextItem)!=NULL)
    {
      /* print word information */
      printf("%d word=%s wordCount=%d docCount=%d IDF=%f\n",i,curItem->text,curItem->wrdCount,curItem->docCount,curItem->IDF);
      /* now print document list */
      d=1;
      while (curDoc != NULL)
	{
	  printf("   %d docName=%s count=%d weight=%f\n",d,curDoc->docName, curDoc->count, curDoc->weight);
	  curDoc=curDoc->nextIndex;
	  d++;
	}
      /* find pointer to next list item */
      curItem=curItem->nextItem;
      /* find pointer to document list for this item */
      curDoc=curItem->docList;
      i++;
    }
  return i;
}

/********************************************************/

/* Function to order the linked list according to word frequency */
int sortList(struct item *root)
{
  int swap;
  int swapCnt;
  struct item *prevIt;
  struct item *curIt;
  struct item *nextIt;

  /* initialise */
  swap=1;
  swapCnt=0;

  /* outer loop - keep going til no more swaps occur */
  while (swap==1)
    {

      /* reset swap flag */
      swap=0;

      /* set pointers to start of list */
      prevIt=root;
      curIt=root->nextItem;
      nextIt=curIt->nextItem;

      while (((curIt->wrdCount) >= (nextIt->wrdCount)) && (nextIt->nextItem)!=NULL)
	{
	  /* move one entry along the linked list */
	  prevIt=curIt;
	  curIt=nextIt;
	  nextIt=curIt->nextItem;
	}
      if ((nextIt->nextItem)!=NULL)
	{
	  struct item *tmp;
	  /* swap items */
	  swap=1;
	  swapCnt++;

	  /* set temporary pointer */
	  tmp=nextIt->nextItem;
	  /* set pointer from previous item */
	  prevIt->nextItem=nextIt;
	  /* set pointer from nextIt (new curIt) */
	  nextIt->nextItem=curIt;
	  /* set pointer from curIt (new nextIt) */
	  curIt->nextItem=tmp;
	}
    }
  return(swapCnt);
}

/********************************************************/

/* Function to create new item in linked list */
void createNewItem
(struct item *prevIt, struct item **newIt, struct item *nextIt, char *wrd, int *listSize, char *docName)
{
  int c;
  /* allocate memory for new linked list item */
  *newIt=calloc(1,sizeof(struct item));
  /* set pointer to new item */
  prevIt->nextItem=*newIt;
  /* set entries in new item */
  (*newIt)->nextItem=nextIt;
  (*newIt)->wrdCount=1;
  (*newIt)->docCount=1;
  (*newIt)->text=calloc(strlen(wrd)+1,sizeof(char));
  for (c=0; c<strlen(wrd); c++)
    {
      (*newIt)->text[c]=wrd[c];
    }
  (*newIt)->text[c]='\0';
  /* now create index for first document which contains new word */
  (*newIt)->docList=calloc(1,sizeof(struct docIndex));
  (*newIt)->docList->count=1;
  (*newIt)->docList->docName=calloc(strlen(docName)+1,sizeof(char));
  for (c=0; c<strlen(docName); c++)
    {
      (*newIt)->docList->docName[c]=docName[c];
    }
  (*newIt)->docList->docName[c]='\0';
  (*newIt)->docList->nextIndex=NULL;
  /* increment listSize */
  (*listSize)++;
}

/********************************************************/

void createNewDocIndex(struct docIndex *docInd,struct docIndex **newIndex,char *documentName)
{
  int c;
  struct docIndex *tmp;
  /* create space for new docIndex item in linked list */
  tmp=calloc(1,sizeof(struct docIndex));
  *newIndex=tmp;
  docInd->nextIndex=tmp;
  tmp->docName=calloc(strlen(documentName)+1,sizeof(char));
  for (c=0; c<strlen(documentName); c++) tmp->docName[c]=documentName[c];
  tmp->docName[c]='\0';
  /* set count */
  tmp->count=1;
  tmp->nextIndex=NULL;
}

/********************************************************/

void calcIDFs(struct item *root)
{
  int i;
  int totDoc;
  struct item *curItem;

  /* initialise */
  i=1;
  totDoc=root->docCount;
  curItem=root->nextItem;
 
  /* go through items in linked list */
  while ((curItem->nextItem)!=NULL)
    {
      float r;
      /* calculate IDF for current word */
      r=totDoc; r=r/(curItem->docCount);
      curItem->IDF=log(r);
      /* find pointer to next list item */
      curItem=curItem->nextItem;
      i++;
    }
}

/********************************************************/

int calcTf_IDF_weights(struct item *root)
{
  int count;
  struct item *curItem;
  struct docIndex *curDoc;

  /* initialise */
  count=0;
  curItem=root->nextItem;
  curDoc=curItem->docList;
  while ((curItem->nextItem)!=NULL)
    {
      float termIDF;
      termIDF=curItem->IDF;
      while (curDoc != NULL)
	{
	  float termFreq;
	  /* set term frequency */
	  termFreq=curDoc->count;
	  /* calculate tf-IDF weight for term i and document d */
	  curDoc->weight=termFreq*termIDF;
	  curDoc=curDoc->nextIndex;
	  /* increment count */
	  count++;
	}
      /* find pointer to next list item */
      curItem=curItem->nextItem;
      /* find pointer to document list for this item */
      curDoc=curItem->docList;
    }
  return count;
}

/********************************************************/

struct docData *calcDocLengths(FILE *scriptFile, int numDocs, struct item *root)
{
  char line[MAX_STR_LEN];
  char docName[MAX_STR_LEN];
  int c;
  int d;
  struct item *curItem;
  struct docIndex *curDoc;
  struct docData *docDat;

  printf("%d\n",numDocs);
  d=0;
  /* create space for document length storage */
  docDat=calloc(numDocs,sizeof(struct docData));

  while (fgets(line,MAX_STR_LEN,scriptFile)!=NULL)
    {
      /* extract file name from string */
      if (sscanf(line,"%s\n",docName)!=1)
	{
	  printf("Error: cannot parse script file line %s\n",line);
	  exit(1);
	}
      /* copy document name into docData array */
      docDat[d].docName=calloc(strlen(docName)+1,sizeof(char));
      for (c=0; c<strlen(docName); c++) docDat[d].docName[c]=docName[c];
      docDat[d].docName[c]='\0';

      /* calculate document length */
      docDat[d].length=0;

      /* now go through the index looking for occurences of this word */

      curItem=root->nextItem;
      curDoc=curItem->docList;
      while ((curItem->nextItem)!=NULL)
	{
	  while (curDoc != NULL)
	    {
	      if (strcmp(docName,curDoc->docName)==0) docDat[d].length+=(curDoc->weight)*(curDoc->weight);
	      curDoc=curDoc->nextIndex;
	    }
	  /* find pointer to next list item */
	  curItem=curItem->nextItem;
	  /* find pointer to document list for this item */
	  curDoc=curItem->docList;
	}
      /* now take square root */
      docDat[d].length=sqrt(docDat[d].length);
      printf("%s %f\n",docDat[d].docName,docDat[d].length);
      /* increment document counter */
      d++;
    }
  printf("\n");
  return docDat;
} 

/********************************************************/

int main(int argc, char *argv[])
{
  char line[MAX_STR_LEN];
  char docName[MAX_STR_LEN];
  char wrd[MAX_STR_LEN];
  char nullText[1];
  FILE *scriptFile;
  FILE *docFile;
  int a;
  int listSize; /* size of linked list */
  int numDocs;
  struct item root;
  struct item end;
  struct item *curItem;
  struct item *prevItem;
  struct item *newItem;
  struct docIndex *docInd;
  struct docIndex *newIndex;
  struct docData *docLen;

  /* set up end of linked list */
  end.wrdCount=0;
  end.docCount;
  nullText[0]='\0';
  end.text=nullText;
  end.nextItem=NULL;
  end.docList=NULL;

  /* initialise linked list counter */
  listSize=0;

  /* Open text file for analysis */
  if (argc!=2)
    {
      printf("format: index scriptFileName\n");
      exit(1);
    }
  /* set pointer to input script file */
  argv++;
  if ((scriptFile=fopen(*argv,"r"))==NULL)
    {
      printf("Error: can't open script file %s\n",*argv);
      exit(1);
    }
  /* Script file exists and is open */
  /* initialise word counter */
  a=0;

  /* initialise document counter */
  numDocs=0;

  /* now cycle though the text files listed in the script file */
  while (fgets(line,MAX_STR_LEN,scriptFile)!=NULL)
    {
      /* extract file name from string */
      if (sscanf(line,"%s\n",docName)!=1)
	{
	  printf("Error: cannot parse script file line %s\n",line);
	  exit(1);
	}
      /* open next text file for analysis */
      if ((docFile=fopen(docName,"r"))==NULL)
	{
	  printf("Error: can't open document file |%s|\n",docName);
	  exit(1);
	}

      /* Now go through the file and read the words */
      numDocs++;
      while (nextWord(docFile,wrd)!=EOF)
	{
	  if (strlen(wrd)>0)
	    {
	      int i;
	      
	      /* increment word counter */
	      a++;
	  
	      if (listSize==0)
		{
		  /* create first proper item in linked list */
		  createNewItem(&root,&newItem,&end,wrd,&listSize,docName);
		}
	      else
		{
		  /* find word in linked list (if it is in the list already) */
		  /* initialise the search to start at beginning of list */
		  i=0;
		  curItem=root.nextItem;
		  prevItem=&root;
		  /* initialisation done */
		  while ((i<listSize)&&(strcmp(wrd,curItem->text)>0))
		    {
		      /* record current item in linked list */
		      prevItem=curItem;
		      /* and move on to next item */
		      curItem=curItem->nextItem;
		      /* increment linked list item counter */
		      i++;
		    }
		  if (strcmp(wrd,curItem->text)==0)
		    {
		      int d;
		      d=1;
		      /* word is already in the list, just increment counter */
		      curItem->wrdCount++;

		      /* see if the current document is already in the document list for this word */
		      docInd=curItem->docList;
		      while ((d<=curItem->docCount) && (strcmp(docName,docInd->docName)!=0))
			{
			  if (d<curItem->docCount) docInd=docInd->nextIndex;
			  d++;
			}
		      /* if (strcmp(docName,docInd->docName)==0) */
		      if (d>curItem->docCount)
			{
			  /* create a new entry in document index for current word */
			  createNewDocIndex(docInd,&newIndex,docName);
			  curItem->docCount++;
			}
		      else
			{
			  /* increment word count for this document */
			  docInd->count++;
			}
		    }
		  else
		    {
		      /* create new item at this point in linked list */
		      createNewItem(prevItem,&newItem,prevItem->nextItem,wrd,&listSize,docName);
		    }
		}
	    }
	}
    }
  /* print something out to reassure user that the program hasn't crashed! */
  /* set root.count to be list size */
  root.wrdCount=listSize;
  root.docCount=numDocs;
  /* sort the list according to word frequency */
  /* sortList(&root);*/
  /* calculate Inverse Document Frequencies */
  calcIDFs(&root);
  /* calculate term - document weights */
  calcTf_IDF_weights(&root);
  /* calculate document lengths */
  rewind(scriptFile);
  docLen=calcDocLengths(scriptFile,numDocs,&root);
  /* print out the sorted linked list */
  checkList(root);
  return(0);
}
